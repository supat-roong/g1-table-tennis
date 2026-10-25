"""Table tennis ball physics, vectorised over environments with NumPy.

MuJoCo simulates the robot; the ball is integrated here and drawn as a mocap
body. Everything works on arrays of shape (N, 3) so one call advances every
environment in a worker.

Flight:   a = g - kd |v| v + km (w x v),   dw/dt = -kw w
Bounces:  an impulse model for a hollow sphere (I = 2/3 m r^2) on a surface
          with restitution e and Coulomb friction mu. The tangential impulse
          either stops the contact point slipping (the ball leaves rolling) or
          saturates at mu * J_n (it leaves still sliding). This is what couples
          spin and bounce: topspin kicks forward off the table, backspin checks.

The paddle rubber is just a surface with its own (e, mu): a sponge has no
analogue in a rigid-body simulator, so different rubbers are modelled as
different coefficients and randomised in training.
"""

from dataclasses import dataclass

import numpy as np

from g1tt import ball_jit
from g1tt.model import NET_H, NET_OVERHANG, TABLE_H, TABLE_LEN, TABLE_WID

RADIUS = 0.02
MASS = 0.0027
GRAVITY = np.array([0.0, 0.0, -9.81])

# Drag: kd = rho * Cd * A / (2m), rho 1.2, Cd 0.45 -> 0.126 1/m.
# Magnus: km = rho * A * r * kL / (2m), lift slope kL ~ 0.5 -> 2.8e-3.
KD = 0.126
KM = 2.8e-3
KW = 0.05

TABLE_E = 0.88  # 30 cm drop rebounds ~23 cm (ITTF)
TABLE_MU = 0.25
RUBBER_E = 0.83
RUBBER_MU = 0.6


@dataclass
class BallParams:
  """Per-environment physical coefficients (arrays of shape (N,))."""
  kd: np.ndarray
  km: np.ndarray
  table_e: np.ndarray
  table_mu: np.ndarray
  rubber_e: np.ndarray
  rubber_mu: np.ndarray

  @classmethod
  def nominal(cls, n: int) -> "BallParams":
    f = lambda x: np.full(n, x, dtype=np.float64)
    return cls(f(KD), f(KM), f(TABLE_E), f(TABLE_MU), f(RUBBER_E), f(RUBBER_MU))

  @classmethod
  def randomised(cls, rng: np.random.Generator, n: int, scale: float = 1.0) -> "BallParams":
    """Domain randomisation around nominal; `scale` 0 gives nominal."""
    u = lambda lo, hi, mid: mid + scale * (rng.uniform(lo, hi, n) - mid)
    return cls(
      kd=u(0.10, 0.15, KD),
      km=u(1.8e-3, 3.8e-3, KM),
      table_e=u(0.85, 0.91, TABLE_E),
      table_mu=u(0.15, 0.35, TABLE_MU),
      rubber_e=u(0.75, 0.90, RUBBER_E),
      rubber_mu=u(0.35, 0.90, RUBBER_MU),
    )

  def take(self, idx) -> "BallParams":
    return BallParams(*(getattr(self, f)[idx] for f in ("kd", "km", "table_e", "table_mu", "rubber_e", "rubber_mu")))

  def put(self, idx, other: "BallParams") -> None:
    for f in ("kd", "km", "table_e", "table_mu", "rubber_e", "rubber_mu"):
      getattr(self, f)[idx] = getattr(other, f)


def cross(a, b):
  """Row-wise cross product; np.cross is several times slower on small arrays."""
  out = np.empty(np.broadcast_shapes(a.shape, b.shape))
  out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
  out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
  out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
  return out


def _accel(v, w, kd, km):
  speed = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
  return GRAVITY - kd[:, None] * speed * v + km[:, None] * cross(w, v)


def flight(p, v, w, h, kd, km):
  """One RK2 (midpoint) step of free flight. Returns new (p, v, w)."""
  a1 = _accel(v, w, kd, km)
  vm = v + 0.5 * h * a1
  a2 = _accel(vm, w, kd, km)
  return p + h * vm, v + h * a2, w * (1.0 - KW * h)


def impact(u, w, n, e, mu):
  """Bounce off a surface.

  u: ball velocity relative to the surface (N, 3); w: spin (N, 3);
  n: unit surface normal pointing towards the ball (N, 3); e, mu: (N,).
  Returns (u', w'). Rows with u.n >= 0 (separating) are returned unchanged.
  """
  un = np.sum(u * n, axis=-1)
  approaching = un < 0
  jn = (1.0 + e) * np.maximum(-un, 0.0)  # normal impulse per unit mass
  rc = -RADIUS * n  # centre -> contact point
  uc = u + cross(w, rc)
  uct = uc - np.sum(uc * n, axis=-1, keepdims=True) * n
  jt_roll = -uct / 2.5  # impulse that ends slip, hollow sphere: 1 + m r^2 / I = 2.5
  mag = np.linalg.norm(jt_roll, axis=-1)
  cap = mu * jn
  scale = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
  jt = jt_roll * scale[:, None]
  inertia = (2.0 / 3.0) * RADIUS**2  # per unit mass
  u_new = u + jn[:, None] * n + jt
  w_new = w + cross(rc, jt) / inertia
  a = approaching[:, None]
  return np.where(a, u_new, u), np.where(a, w_new, w)


def on_table(x, y):
  return (np.abs(x) <= TABLE_LEN / 2) & (np.abs(y) <= TABLE_WID / 2)


# Event codes for a ball tick.
EV_NONE, EV_TABLE, EV_NET, EV_FLOOR, EV_PADDLE = 0, 1, 2, 3, 4


def step_world(p, v, w, h, prm: BallParams, alive):
  """Advance free flight by h with table, net and floor contacts.

  Returns (p, v, w, table_hit, net_hit, floor_hit), each event a bool (N,).
  `alive` masks balls in play; dead balls are left untouched.
  """
  p1, v1, w1 = flight(p, v, w, h, prm.kd, prm.km)
  zc = TABLE_H + RADIUS

  # Table: the centre crosses the surface plus a radius, moving down, over the table.
  table = alive & (p[:, 2] >= zc) & (p1[:, 2] < zc) & on_table(p1[:, 0], p1[:, 1]) & (v1[:, 2] < 0)
  if table.any():
    n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), p1.shape)
    vb, wb = impact(v1, w1, n, prm.table_e, prm.table_mu)
    v1 = np.where(table[:, None], vb, v1)
    w1 = np.where(table[:, None], wb, w1)
    p1[table, 2] = 2 * zc - p1[table, 2]

  # Net: a crossing of x = 0 within net height and width.
  cross = (np.sign(p[:, 0]) != np.sign(p1[:, 0]))
  net = alive & cross & (p1[:, 2] - RADIUS < TABLE_H + NET_H) & (np.abs(p1[:, 1]) < TABLE_WID / 2 + NET_OVERHANG)

  floor = alive & (p1[:, 2] < RADIUS)

  keep = ~alive
  p1 = np.where(keep[:, None], p, p1)
  v1 = np.where(keep[:, None], v, v1)
  w1 = np.where(keep[:, None], w, w1)
  return p1, v1, w1, table, net, floor


def paddle_contact(p0, p1, v1, w1, c0, c1, n0, n1, vpad, wpad, radius, can_hit, prm: BallParams):
  """Swept ball-blade test over one tick and the bounce it causes.

  The ball goes p0 -> p1 and the blade centre c0 -> c1 with face normal
  n0 -> n1 (both linear over the tick). A hit is a change of side of the
  blade plane, or ending within a ball radius of it while approaching, with
  the crossing point inside the blade disc. Testing the plane crossing,
  rather than overlap at the tick end, keeps a 30 m/s ball from passing
  through a 1 cm blade.

  vpad, wpad: blade linear and angular velocity (N, 3) for the surface speed.
  Returns (p, v, w, hit, side) with side +1 on the face n points to, -1 on the other.
  """
  d0 = np.sum((p0 - c0) * n0, axis=-1)
  d1 = np.sum((p1 - c1) * n1, axis=-1)
  side0 = np.where(d0 >= 0, 1.0, -1.0)
  crossed = np.sign(d0) != np.sign(d1)
  close = np.abs(d1) < RADIUS
  # Fraction of the tick at which the plane is crossed (or end of tick).
  alpha = np.where(crossed, d0 / np.where(crossed, d0 - d1, 1.0), 1.0)
  alpha = np.clip(alpha, 0.0, 1.0)[:, None]
  pa = p0 + alpha * (p1 - p0)
  ca = c0 + alpha * (c1 - c0)
  na = n0 + alpha * (n1 - n0)
  na = na / np.linalg.norm(na, axis=-1, keepdims=True)
  rel = pa - ca
  inplane = rel - np.sum(rel * na, axis=-1, keepdims=True) * na
  inside = np.linalg.norm(inplane, axis=-1) < radius + 0.5 * RADIUS

  face = side0[:, None] * na  # normal on the ball's side
  surf_v = vpad + cross(wpad, rel)
  u = v1 - surf_v
  approaching = np.sum(u * face, axis=-1) < 0
  hit = can_hit & inside & approaching & (crossed | close)
  if not hit.any():
    return p1, v1, w1, hit, side0

  ub, wb = impact(u, w1, face, prm.rubber_e, prm.rubber_mu)
  vb = ub + surf_v
  # Put the ball back on its side of the blade, one radius out.
  pb = ca + inplane + face * (RADIUS + 1e-3)
  h = hit[:, None]
  return np.where(h, pb, p1), np.where(h, vb, v1), np.where(h, wb, w1), hit, side0


def predict(p, v, w, prm: BallParams, horizon: float, h: float = 0.004):
  """Roll the ball forward with table bounces only (no paddle).

  Returns positions, velocities and spins of shape (T, N, 3) and the number
  of table bounces so far at each sample, (T, N).
  """
  if ball_jit.ENABLED:
    return ball_jit.predict(p, v, w, prm, horizon, h)
  steps = int(round(horizon / h))
  n = p.shape[0]
  P = np.empty((steps, n, 3))
  V = np.empty((steps, n, 3))
  W = np.empty((steps, n, 3))
  B = np.empty((steps, n), dtype=np.int32)
  alive = np.ones(n, dtype=bool)
  bounces = np.zeros(n, dtype=np.int32)
  for k in range(steps):
    p, v, w, table, net, floor = step_world(p, v, w, h, prm, alive)
    bounces += table
    P[k], V[k], W[k], B[k] = p, v, w, bounces
  return P, V, W, B


def solve_launch(p0, target, t_flight, w, prm: BallParams, iters: int = 5, h: float = 0.005):
  """Initial velocity that carries a ball from p0 to `target` in `t_flight`
  under drag and Magnus (shooting with fixed-point correction)."""
  if ball_jit.ENABLED:
    return ball_jit.solve_launch(p0, target, t_flight, w, prm, iters, h)
  t = t_flight[:, None]
  v0 = (target - p0) / t - 0.5 * GRAVITY * t
  steps = np.maximum(1, np.round(t_flight / h).astype(int))
  for _ in range(iters):
    p, v, ww = p0.copy(), v0.copy(), w.copy()
    end = np.empty_like(p0)
    for k in range(steps.max()):
      p, v, ww = flight(p, v, ww, h, prm.kd, prm.km)
      done_now = steps == k + 1
      end[done_now] = p[done_now]
    v0 = v0 + (target - end) / t
  return v0


def return_shortfall(p, v, w, prm: BallParams, horizon: float = 1.5, h: float = 0.004):
  """How far a just-hit ball falls short of a legal return, in metres.

  0 for a ball that clears the net and first lands on the opponent's half
  (x > 0). Otherwise the height it lacks at the net, or the distance from
  where it first comes down to the opponent's half. A graded version of
  "did it land", available at the moment of the hit.
  """
  if ball_jit.ENABLED:
    return ball_jit.return_shortfall(p, v, w, prm, horizon, h)
  n = p.shape[0]
  alive = np.ones(n, dtype=bool)
  short = np.zeros(n)
  done = np.zeros(n, dtype=bool)
  net_top = TABLE_H + NET_H + RADIUS
  for _ in range(int(horizon / h)):
    p0 = p
    p, v, w = flight(p, v, w, h, prm.kd, prm.km)
    # Net plane crossing (or hitting it while still on our side).
    cross = ~done & (p0[:, 0] < 0) & (p[:, 0] >= 0)
    lack = np.maximum(net_top - p[:, 2], 0.0)
    over_net_width = np.abs(p[:, 1]) < TABLE_WID / 2 + NET_OVERHANG
    hit_net = cross & over_net_width & (lack > 0)
    short = np.where(hit_net, lack, short)
    done |= hit_net
    # Coming down to table height: score where.
    down = ~done & (p0[:, 2] >= TABLE_H + RADIUS) & (p[:, 2] < TABLE_H + RADIUS)
    x, y = p[:, 0], p[:, 1]
    dx = np.maximum(np.maximum(0.02 - x, x - TABLE_LEN / 2), 0.0)
    dy = np.maximum(np.abs(y) - TABLE_WID / 2, 0.0)
    short = np.where(down, np.hypot(dx, dy), short)
    done |= down
    if done.all():
      break
  # Never came down within the horizon: far long.
  short = np.where(done, short, np.maximum(p[:, 0] - TABLE_LEN / 2, 1.0))
  return short
