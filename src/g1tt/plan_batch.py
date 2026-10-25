"""The hit planner as one compiled call over every ball at once (GPU-ready).

g1tt.planner.plan_hits plans a group of balls with one aim, and decides in
Python which ones get new landing spots; g1tt.duel._plan_for then re-plans
backhands with more net clearance. Here the same rules run as fixed-size,
masked array work, so a single jax.jit call covers balls with different aims,
landing ranges (SGS cells) and retry counts, and could cover thousands of
matches on a GPU:

- predict each ball to where it crosses the hitting plane after the bounces
  still to come, keeping only that crossing (no trajectory);
- draw the first landing spot and `max_retries` more per ball with JAX random
  numbers, inside that ball's own ranges; retries past a ball's own count are
  masked off;
- solve every candidate return in one batch, for both net clearances (the
  ball's own and the backhand one with its deeper landing);
- per ball take the first candidate whose return is valid and reaches the
  partner (planner.plan_hits' retry rule), else the first; then use the
  backhand-clearance result for valid backhands, as _plan_for does.

Random numbers differ from the NumPy planner's, so results agree in
distribution rather than ball by ball; with a fixed landing spot and no
retries they agree to float32 precision.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from g1tt import ball_jit, planner
from g1tt.ball_jit import _bucket, _pad, _return_core, _step_world
from g1tt.reference import RefLib

BALL_Z = planner.BALL_Z
FIELDS = ("valid", "t_hit", "ball_pos", "ball_vel_in", "ball_vel_out", "blade_pos", "normal", "blade_vel", "land")


def _first_plane(p, v, w, kd, km, te, tmu, h, x_plane, bounces, steps_arr):
  """First predicted sample after `bounces` table bounces with the ball at or
  past x_plane moving -x: (found, k, position, velocity, spin)."""
  def body(carry, k):
    p, v, w, b, found, kk, pos, vel, spin = carry
    p, v, w, table = _step_world(p, v, w, h, kd, km, te, tmu)
    b = b + table.astype(jnp.int32)
    at = (b == bounces) & (p[:, 0] <= x_plane) & (v[:, 0] < 0) & ~found
    pos = jnp.where(at[:, None], p, pos)
    vel = jnp.where(at[:, None], v, vel)
    spin = jnp.where(at[:, None], w, spin)
    kk = jnp.where(at, k, kk)
    return (p, v, w, b, found | at, kk, pos, vel, spin), None
  n = p.shape[0]
  z3 = jnp.zeros_like(p)
  init = (p, v, w, jnp.zeros(n, jnp.int32), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32), z3, z3, z3)
  (_, _, _, _, found, k, pos, vel, spin), _ = jax.lax.scan(body, init, steps_arr)
  return found, k, pos, vel, spin


@functools.partial(jax.jit, static_argnames=("bounces", "strict", "ts", "prefer", "band", "reach_steps",
                                             "max_retries", "fixed_land"))
def _plan(key, p, v, w, kd, km, te, tmu, re, rmu,
          land_x, land_y, py, retries, clear, bh_on, bh_clear, bh_land_x, land_fixed,
          hit_x, hit_y0, hit_y1, hit_z0, hit_z1, pz0, pz1, from_y, h, steps_arr,
          bounces, strict, ts, prefer, band, reach_steps, max_retries, fixed_land):
  n = p.shape[0]
  found, k, bp, bv, bw = _first_plane(p, v, w, kd, km, te, tmu, h, hit_x, bounces, steps_arr)
  t_hit = (k + 1) * h
  valid = found
  if strict:
    valid = valid & (bp[:, 2] > hit_z0) & (bp[:, 2] < hit_z1) & (bp[:, 1] > hit_y0) & (bp[:, 1] < hit_y1)

  # Candidate landing spots: r = 0 is the first draw, 1..max_retries the retries;
  # variant 0 uses the ball's own depth, variant 1 the backhand depth.
  r1 = max_retries + 1
  kx, ky, kb = jax.random.split(key, 3)
  u = lambda k_, lo_hi, shape: lo_hi[:, 0] + (lo_hi[:, 1] - lo_hi[:, 0]) * jax.random.uniform(k_, shape, dtype=p.dtype)
  y_c = u(ky, land_y, (r1, n))
  x_c = u(kx, land_x, (r1, n))
  x_b = u(kb, bh_land_x, (r1, n))
  if fixed_land:
    x_c = x_c.at[0].set(land_fixed[:, 0])
    y_c = y_c.at[0].set(land_fixed[:, 1])
    x_b = x_b.at[0].set(land_fixed[:, 0])
  zc = jnp.full((r1, n), BALL_Z, dtype=p.dtype)
  lands = jnp.stack([jnp.stack([x_c, y_c, zc], -1), jnp.stack([x_b, y_c, zc], -1)])  # (2, r1, n, 3)
  clears = jnp.stack([clear, jnp.maximum(clear, bh_clear)])  # (2, n)

  # One batch: every variant x candidate x ball.
  m = 2 * r1
  tile = lambda x: jnp.tile(x, (m,) + (1,) * (x.ndim - 1))
  out = _return_core(tile(bp), tile(bv), tile(bw), lands.reshape(m * n, 3), *map(tile, (kd, km, te, tmu, re, rmu)),
                     hit_x, pz0, pz1, tile(py[:, 0]), tile(py[:, 1]), jnp.repeat(clears, r1, axis=0).reshape(m * n),
                     ts, prefer, band, reach_steps)
  v_out, ok, normal, blade_vel = (x.reshape((2, r1, n) + x.shape[1:]) for x in out)

  # Retry rule (planner.plan_hits): first candidate whose return is valid and
  # reaches the partner (judged with no spin); balls with no hit to make keep
  # their first spot; retries past the ball's own count do not exist.
  if band and max_retries > 0 and not fixed_land:
    zero_w = jnp.zeros((m * n, 3), dtype=p.dtype)
    fnd, pos = ball_jit._first_crossing(tile(bp), v_out.reshape(m * n, 3), zero_w, *map(tile, (kd, km, te, tmu)),
                                        0.006, -hit_x, 1, jnp.zeros(reach_steps))
    yt = -pos[:, 1]
    reach = (fnd & (pos[:, 2] > pz0) & (pos[:, 2] < pz1) & (yt > tile(py[:, 0]) + 0.05)
             & (yt < tile(py[:, 1]) - 0.05)).reshape(2, r1, n)
    good = ok & reach
    good = good.at[:, 0].set(good[:, 0] | ~valid[None])
    good = good & (jnp.arange(r1)[None, :, None] <= retries[None, None, :])
    pick = jnp.where(good.any(axis=1), jnp.argmax(good, axis=1), 0)  # (2, n)
  else:
    pick = jnp.zeros((2, n), jnp.int32)
  sel = lambda x: jnp.take_along_axis(x, pick.reshape((2, 1, n) + (1,) * (x.ndim - 3)), axis=1)[:, 0]
  v_out, ok, normal, blade_vel, land = (sel(x) for x in (v_out, ok, normal, blade_vel, lands))

  # Backhand clearance (duel._plan_for): valid backhands under the ball's own
  # clearance are re-planned higher; keep that where it is still a valid backhand.
  blade = bp[None] - normal * ball_jit.RADIUS
  valid2 = valid[None] & ok
  is_bh = blade[..., 1] > from_y
  use_hi = bh_on & (bh_clear > clear) & valid2[0] & is_bh[0] & valid2[1] & is_bh[1]
  choose = lambda x: jnp.where(use_hi.reshape((n,) + (1,) * (x.ndim - 2)), x[1], x[0])
  return (choose(valid2), t_hit, bp, bv, choose(v_out), choose(blade), choose(normal), choose(blade_vel), choose(land))


def plan_batch(p, v, w, prm, key, *, land_x, land_y, partner_y, retries, clear, cfg: planner.PlanCfg,
               bounces_left=1, strict=True, horizon=1.6, bh_clear=None, bh_land_x=None,
               land=None, max_retries=None, reach_horizon=1.3) -> planner.Plan:
  """Plan every ball at once. land_x, land_y, partner_y: (n, 2) ranges per ball;
  retries, clear: (n,) per ball. bh_clear / bh_land_x: the backhand re-plan
  (None for none).
  land: fixed first landing spots (n, 3) and no retries."""
  n = p.shape[0]
  b = _bucket(n)
  pad = lambda x: _pad(x, b)
  max_retries = int(np.max(retries)) if max_retries is None else max_retries
  if land is not None:
    max_retries = 0
  steps = int(round(horizon / 0.004))
  bh_on = bh_clear is not None
  out = _plan(key, pad(p), pad(v), pad(w), *(pad(getattr(prm, f)) for f in
              ("kd", "km", "table_e", "table_mu", "rubber_e", "rubber_mu")),
              pad(land_x), pad(land_y), pad(partner_y), _pad(np.asarray(retries, dtype=np.float32), b).astype(np.int32),
              pad(clear), bh_on, float(bh_clear or 0.0),
              pad(np.tile(bh_land_x if bh_land_x is not None else (0.0, 0.0), (n, 1))) if bh_on and bh_land_x is not None
              else pad(land_x), pad(land if land is not None else np.zeros((n, 3))),
              cfg.hit_x, cfg.hit_y[0], cfg.hit_y[1], cfg.hit_z[0], cfg.hit_z[1], cfg.partner_z[0], cfg.partner_z[1],
              RefLib.BACKHAND_FROM_Y, horizon / steps, jnp.arange(steps),
              bounces=bounces_left, strict=strict, ts=tuple(float(t) for t in cfg.return_t),
              prefer=None if cfg.prefer_out_speed is None else float(cfg.prefer_out_speed),
              band=bool(cfg.partner_band), reach_steps=int(round(reach_horizon / 0.006)),
              max_retries=max_retries, fixed_land=land is not None)
  arrs = [np.array(x[:n]) for x in out]
  arrs[0] = arrs[0].astype(bool)
  return planner.Plan(*arrs)
