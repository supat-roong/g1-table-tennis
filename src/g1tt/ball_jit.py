"""Compiled (JAX, CPU) versions of the ball computations the planner repeats.

The planner rolls balls forward step by step for hundreds of steps on a few
balls at a time; in NumPy that is almost all Python overhead (~80 us a step
whatever the batch), and it was 92% of a self-play training step. These
functions are the same maths as g1tt.ball / g1tt.planner, compiled with
jax.jit and lax.scan.

Precision is float32, as a GPU would run it: plans agree with the float64
NumPy versions to ~1e-6 m/s and ~0.03 deg of paddle face, with the same
verdicts and choices (400 rally returns checked). G1TT_JAX_X64=1 gives float64,
matching NumPy to ~1e-15, for replaying runs made before float32.

Batches are padded to a few fixed sizes so each function compiles a handful of
times. JAX runs single-threaded here: training already runs one worker process
per core. Set G1TT_NUMPY_BALL=1 to use the NumPy versions instead.
"""

import functools
import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax  # noqa: E402

X64 = os.environ.get("G1TT_JAX_X64", "0") == "1"
jax.config.update("jax_enable_x64", X64)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from g1tt.model import NET_H, NET_OVERHANG, TABLE_H, TABLE_LEN, TABLE_WID  # noqa: E402

ENABLED = os.environ.get("G1TT_NUMPY_BALL", "") != "1"

RADIUS = 0.02
GRAVITY = jnp.array([0.0, 0.0, -9.81])
KW = 0.05
_BUCKETS = (8, 32, 128, 512, 2048, 8192)


def _bucket(n: int) -> int:
  for b in _BUCKETS:
    if n <= b:
      return b
  return int(2 ** np.ceil(np.log2(n)))


def _pad(x, size):
  x = np.asarray(x, dtype=np.float64 if X64 else np.float32)
  if x.shape[0] == size:
    return x
  pad = np.zeros((size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
  if x.ndim == 1:
    pad[:] = x[0] if x.shape[0] else 0.0  # keep coefficients sane in padded rows
  return np.concatenate([x, pad])


# ---------------------------------------------------------------- kernels
def _accel(v, w, kd, km):
  speed = jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True))
  return GRAVITY - kd[:, None] * speed * v + km[:, None] * jnp.cross(w, v)


def _flight(p, v, w, h, kd, km):
  a1 = _accel(v, w, kd, km)
  vm = v + 0.5 * h * a1
  a2 = _accel(vm, w, kd, km)
  return p + h * vm, v + h * a2, w * (1.0 - KW * h)


def _table_impact(u, w, e, mu):
  n = jnp.array([0.0, 0.0, 1.0])
  un = u[:, 2]
  jn = (1.0 + e) * jnp.maximum(-un, 0.0)
  rc = -RADIUS * n
  uc = u + jnp.cross(w, rc)
  uct = uc.at[:, 2].set(0.0)
  jt_roll = -uct / 2.5
  mag = jnp.linalg.norm(jt_roll, axis=-1)
  cap = mu * jn
  scale = jnp.where(mag > cap, cap / jnp.maximum(mag, 1e-12), 1.0)
  jt = jt_roll * scale[:, None]
  inertia = (2.0 / 3.0) * RADIUS ** 2
  u_new = u + jn[:, None] * n + jt
  w_new = w + jnp.cross(rc, jt) / inertia
  a = (un < 0)[:, None]
  return jnp.where(a, u_new, u), jnp.where(a, w_new, w)


def _step_world(p, v, w, h, kd, km, te, tmu):
  """ball.step_world with every ball alive, table contacts only (what predict uses)."""
  p1, v1, w1 = _flight(p, v, w, h, kd, km)
  zc = TABLE_H + RADIUS
  on = (jnp.abs(p1[:, 0]) <= TABLE_LEN / 2) & (jnp.abs(p1[:, 1]) <= TABLE_WID / 2)
  table = (p[:, 2] >= zc) & (p1[:, 2] < zc) & on & (v1[:, 2] < 0)
  vb, wb = _table_impact(v1, w1, te, tmu)
  v1 = jnp.where(table[:, None], vb, v1)
  w1 = jnp.where(table[:, None], wb, w1)
  p1 = p1.at[:, 2].set(jnp.where(table, 2 * zc - p1[:, 2], p1[:, 2]))
  return p1, v1, w1, table


@jax.jit
def _predict(p, v, w, kd, km, te, tmu, h, steps_arr):
  def body(carry, _):
    p, v, w, b = carry
    p, v, w, table = _step_world(p, v, w, h, kd, km, te, tmu)
    b = b + table.astype(jnp.int32)
    return (p, v, w, b), (p, v, w, b)
  _, out = jax.lax.scan(body, (p, v, w, jnp.zeros(p.shape[0], jnp.int32)), steps_arr)
  return out


@jax.jit
def _solve_launch(p0, target, t_flight, w, kd, km, h, iters_arr, steps_arr):
  t = t_flight[:, None]
  v0 = (target - p0) / t - 0.5 * GRAVITY * t
  steps = jnp.maximum(1, jnp.round(t_flight / h).astype(jnp.int32))

  def one(v0, _):
    def body(carry, k):
      p, v, ww, end = carry
      p, v, ww = _flight(p, v, ww, h, kd, km)
      end = jnp.where((steps == k + 1)[:, None], p, end)
      return (p, v, ww, end), None
    (_, _, _, end), _ = jax.lax.scan(body, (p0, v0, w, p0), steps_arr)
    return v0 + (target - end) / t, None

  v0, _ = jax.lax.scan(one, v0, iters_arr)
  return v0


@jax.jit
def _net_height(p, v, w, kd, km, h, steps_arr):
  """Height of the ball centre where it crosses x = 0 (-inf if it does not), as in planner._solve_return."""
  def body(carry, _):
    p, v, w, z = carry
    pn, vn, wn = _flight(p, v, w, h, kd, km)
    crossing = (p[:, 0] < 0) & (pn[:, 0] >= 0)
    f = -p[:, 0] / jnp.where(crossing, pn[:, 0] - p[:, 0], 1.0)
    z = jnp.where(crossing, p[:, 2] + f * (pn[:, 2] - p[:, 2]), z)
    return (pn, vn, wn, z), None
  (_, _, _, z), _ = jax.lax.scan(body, (p, v, w, jnp.full(p.shape[0], -jnp.inf)), steps_arr)
  return z


@jax.jit
def _first_crossing(p, v, w, kd, km, te, tmu, h, x_plane, bounces, steps_arr):
  """First predicted sample after `bounces` table bounces where the ball has
  passed x_plane moving +x: (found, position). planner._partner_ok's test without
  keeping the whole trajectory."""
  def body(carry, _):
    p, v, w, b, found, pos = carry
    p, v, w, table = _step_world(p, v, w, h, kd, km, te, tmu)
    b = b + table.astype(jnp.int32)
    at = (b == bounces) & (p[:, 0] >= x_plane) & (v[:, 0] > 0) & ~found
    pos = jnp.where(at[:, None], p, pos)
    return (p, v, w, b, found | at, pos), None
  n = p.shape[0]
  init = (p, v, w, jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), jnp.zeros((n, 3)))
  (_, _, _, _, found, pos), _ = jax.lax.scan(body, init, steps_arr)
  return found, pos


# ---------------------------------------------------------------- NumPy-facing wrappers
def predict(p, v, w, prm, horizon: float, h: float):
  n = p.shape[0]
  steps = int(round(horizon / h))
  b = _bucket(n)
  P, V, W, B = _predict(_pad(p, b), _pad(v, b), _pad(w, b), _pad(prm.kd, b), _pad(prm.km, b),
                        _pad(prm.table_e, b), _pad(prm.table_mu, b), h, jnp.zeros(steps))
  return (np.asarray(P)[:, :n], np.asarray(V)[:, :n], np.asarray(W)[:, :n], np.asarray(B)[:, :n])


def solve_launch(p0, target, t_flight, w, prm, iters: int, h: float):
  n = p0.shape[0]
  max_steps = int(np.ceil(max(1, int(np.round(np.max(t_flight) / h))) / 10) * 10)
  b = _bucket(n)
  tf = _pad(t_flight, b)
  out = _solve_launch(_pad(p0, b), _pad(target, b), tf, _pad(w, b), _pad(prm.kd, b), _pad(prm.km, b),
                      h, jnp.zeros(iters), jnp.arange(max_steps))
  return np.asarray(out)[:n]


def net_height(p, v, w, prm, h: float, steps: int):
  n = p.shape[0]
  b = _bucket(n)
  z = _net_height(_pad(p, b), _pad(v, b), _pad(w, b), _pad(prm.kd, b), _pad(prm.km, b), h, jnp.zeros(steps))
  return np.asarray(z)[:n]


def first_crossing(p, v, w, prm, x_plane: float, bounces: int, horizon: float, h: float):
  n = p.shape[0]
  b = _bucket(n)
  found, pos = _first_crossing(_pad(p, b), _pad(v, b), _pad(w, b), _pad(prm.kd, b), _pad(prm.km, b),
                               _pad(prm.table_e, b), _pad(prm.table_mu, b), h, x_plane, bounces,
                               jnp.zeros(int(round(horizon / h))))
  return np.asarray(found)[:n], np.asarray(pos)[:n]


# ---------------------------------------------------------------- paddle, shortfall, whole return solve
def _impact(u, w, n, e, mu):
  """ball.impact: bounce off a surface with unit normal n (per ball)."""
  un = jnp.sum(u * n, axis=-1)
  jn = (1.0 + e) * jnp.maximum(-un, 0.0)
  rc = -RADIUS * n
  uc = u + jnp.cross(w, rc)
  uct = uc - jnp.sum(uc * n, axis=-1, keepdims=True) * n
  jt_roll = -uct / 2.5
  mag = jnp.linalg.norm(jt_roll, axis=-1)
  cap = mu * jn
  scale = jnp.where(mag > cap, cap / jnp.maximum(mag, 1e-12), 1.0)
  jt = jt_roll * scale[:, None]
  inertia = (2.0 / 3.0) * RADIUS ** 2
  u_new = u + jn[:, None] * n + jt
  w_new = w + jnp.cross(rc, jt) / inertia
  a = (un < 0)[:, None]
  return jnp.where(a, u_new, u), jnp.where(a, w_new, w)


def _invert(v_in, w_in, v_out, e, mu, iters: int):
  """planner.invert_paddle."""
  aim = v_out
  for _ in range(iters):
    d = aim - v_in
    n = d / jnp.maximum(jnp.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    vpn = (jnp.sum(aim * n, -1) + e * jnp.sum(v_in * n, -1)) / (1.0 + e)
    vp = vpn[:, None] * n
    u, _ = _impact(v_in - vp, w_in, n, e, mu)
    aim = aim + (v_out - (u + vp))
  return n, vp


@jax.jit
def _shortfall(p, v, w, kd, km, h, steps_arr):
  net_top = TABLE_H + NET_H + RADIUS

  def body(c, _):
    p, v, w, short, done = c
    p0 = p
    p, v, w = _flight(p, v, w, h, kd, km)
    cross = ~done & (p0[:, 0] < 0) & (p[:, 0] >= 0)
    lack = jnp.maximum(net_top - p[:, 2], 0.0)
    hit_net = cross & (jnp.abs(p[:, 1]) < TABLE_WID / 2 + NET_OVERHANG) & (lack > 0)
    short = jnp.where(hit_net, lack, short)
    done = done | hit_net
    down = ~done & (p0[:, 2] >= TABLE_H + RADIUS) & (p[:, 2] < TABLE_H + RADIUS)
    x, y = p[:, 0], p[:, 1]
    dx = jnp.maximum(jnp.maximum(0.02 - x, x - TABLE_LEN / 2), 0.0)
    dy = jnp.maximum(jnp.abs(y) - TABLE_WID / 2, 0.0)
    short = jnp.where(down, jnp.hypot(dx, dy), short)
    return (p, v, w, short, done | down), None

  n = p.shape[0]
  (p, _, _, short, done), _ = jax.lax.scan(body, (p, v, w, jnp.zeros(n), jnp.zeros(n, bool)), steps_arr)
  return jnp.where(done, short, jnp.maximum(p[:, 0] - TABLE_LEN / 2, 1.0))


def _return_core(bp, bv, bw, land, kd, km, te, tmu, re, rmu, hit_x, pz0, pz1, py0, py1, clear,
                 ts, prefer, band, reach_steps):
  """planner._return_for (two rounds of _solve_return, invert_paddle, impact).

  Traceable: net clearance (clear) and the partner's y band (py0, py1) are per
  ball, so callers can mix aims and strokes in one batch.
  """
  n, c = bp.shape[0], len(ts)
  rep = lambda x: jnp.tile(x, (c,) + (1,) * (x.ndim - 1))
  P, L = rep(bp), rep(land)
  tt = jnp.repeat(jnp.asarray(ts, dtype=bp.dtype), n)
  kdr, kmr, ter, tmur, rer, rmur, vin, win = map(rep, (kd, km, te, tmu, re, rmu, bv, bw))
  clr, py0r, py1r = rep(clear), rep(py0), rep(py1)
  launch_steps = jnp.arange(int(np.ceil(max(1, int(round(max(ts) / 0.01))) / 10) * 10))
  net_steps = jnp.zeros(int(max(ts) / 0.01) + 1)
  idx = jnp.arange(n)
  w_out = jnp.zeros_like(bp)
  for _ in range(2):
    Wo = rep(w_out)
    v = _solve_launch(P, L, tt, Wo, kdr, kmr, 0.01, jnp.zeros(3), launch_steps)
    ok = _net_height(P, v, Wo, kdr, kmr, 0.01, net_steps) - RADIUS > TABLE_H + NET_H + clr
    if prefer is not None:
      cost = jnp.where(ok, jnp.abs(jnp.linalg.norm(v, axis=-1) - prefer), jnp.inf)
    else:
      _, vp = _invert(vin, win, v, rer, rmur, 3)
      cost = jnp.where(ok, jnp.linalg.norm(vp, axis=-1), jnp.inf)
    if band:
      found, pos = _first_crossing(P, v, Wo, kdr, kmr, ter, tmur, 0.006, -hit_x, 1, jnp.zeros(reach_steps))
      y = -pos[:, 1]
      reach = found & (pos[:, 2] > pz0) & (pos[:, 2] < pz1) & (y > py0r + 0.05) & (y < py1r - 0.05)
      cost = cost + jnp.where(reach, 0.0, 100.0)
    cost = cost.reshape(c, n)
    best = jnp.argmin(cost, axis=0)
    v_out = v[best * n + idx]
    ok_out = jnp.isfinite(cost[best, idx])
    normal, blade_vel = _invert(bv, bw, v_out, re, rmu, 6)
    _, w_out = _impact(bv - blade_vel, bw, normal, re, rmu)
  return v_out, ok_out, normal, blade_vel


@functools.partial(jax.jit, static_argnames=("ts", "prefer", "band", "reach_steps"))
def _return_for(bp, bv, bw, land, kd, km, te, tmu, re, rmu, hit_x, pz0, pz1, py0, py1, clear,
                ts, prefer, band, reach_steps):
  return _return_core(bp, bv, bw, land, kd, km, te, tmu, re, rmu, hit_x, pz0, pz1, py0, py1, clear,
                      ts, prefer, band, reach_steps)


def return_shortfall(p, v, w, prm, horizon: float, h: float):
  n = p.shape[0]
  b = _bucket(n)
  out = _shortfall(_pad(p, b), _pad(v, b), _pad(w, b), _pad(prm.kd, b), _pad(prm.km, b), h,
                   jnp.zeros(int(horizon / h)))
  return np.asarray(out)[:n]


def return_for(bp, bv, bw, land, prm, cfg, reach_horizon: float = 1.3):
  n = bp.shape[0]
  b = _bucket(n)
  pad = lambda x: _pad(x, b)
  out = _return_for(pad(bp), pad(bv), pad(bw), pad(land), *(pad(getattr(prm, f)) for f in
                    ("kd", "km", "table_e", "table_mu", "rubber_e", "rubber_mu")),
                    cfg.hit_x, cfg.partner_z[0], cfg.partner_z[1], np.full(b, cfg.partner_y[0]),
                    np.full(b, cfg.partner_y[1]), np.full(b, cfg.net_clearance),
                    ts=tuple(float(t) for t in cfg.return_t),
                    prefer=None if cfg.prefer_out_speed is None else float(cfg.prefer_out_speed),
                    band=bool(cfg.partner_band), reach_steps=int(round(reach_horizon / 0.006)))
  return tuple(np.array(x[:n]) for x in out)  # writable: plan_hits updates them in place
