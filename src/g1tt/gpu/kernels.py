"""JAX versions of the per-tick ball and stroke-reference maths the duel uses:
g1tt.ball.step_world / paddle_contact and g1tt.reference.RefLib lookups.

Everything works on whole batches with masks instead of Python branches, so
it runs inside one compiled environment step.
"""

import jax
import jax.numpy as jnp
import numpy as np

from g1tt.ball_jit import KW, RADIUS, _flight, _impact
from g1tt.model import NET_H, NET_OVERHANG, TABLE_H, TABLE_LEN, TABLE_WID


# ---------------------------------------------------------------- ball
def step_world(p, v, w, h, kd, km, te, tmu, alive):
  """ball.step_world: free flight with table, net and floor contacts."""
  p1, v1, w1 = _flight(p, v, w, h, kd, km)
  zc = TABLE_H + RADIUS
  on = (jnp.abs(p1[:, 0]) <= TABLE_LEN / 2) & (jnp.abs(p1[:, 1]) <= TABLE_WID / 2)
  table = alive & (p[:, 2] >= zc) & (p1[:, 2] < zc) & on & (v1[:, 2] < 0)
  n = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0], dtype=p.dtype), p1.shape)
  vb, wb = _impact(v1, w1, n, te, tmu)
  v1 = jnp.where(table[:, None], vb, v1)
  w1 = jnp.where(table[:, None], wb, w1)
  p1 = p1.at[:, 2].set(jnp.where(table, 2 * zc - p1[:, 2], p1[:, 2]))
  cross = jnp.sign(p[:, 0]) != jnp.sign(p1[:, 0])
  net = alive & cross & (p1[:, 2] - RADIUS < TABLE_H + NET_H) & (jnp.abs(p1[:, 1]) < TABLE_WID / 2 + NET_OVERHANG)
  floor = alive & (p1[:, 2] < RADIUS)
  keep = ~alive[:, None]
  return jnp.where(keep, p, p1), jnp.where(keep, v, v1), jnp.where(keep, w, w1), table, net, floor


def paddle_contact(p0, p1, v1, w1, c0, c1, n0, n1, vpad, wpad, radius, can_hit, re, rmu):
  """ball.paddle_contact: swept ball-blade test over one tick and the bounce it causes."""
  d0 = jnp.sum((p0 - c0) * n0, axis=-1)
  d1 = jnp.sum((p1 - c1) * n1, axis=-1)
  side0 = jnp.where(d0 >= 0, 1.0, -1.0).astype(p0.dtype)
  crossed = jnp.sign(d0) != jnp.sign(d1)
  close = jnp.abs(d1) < RADIUS
  alpha = jnp.where(crossed, d0 / jnp.where(crossed, d0 - d1, 1.0), 1.0)
  alpha = jnp.clip(alpha, 0.0, 1.0)[:, None]
  pa = p0 + alpha * (p1 - p0)
  ca = c0 + alpha * (c1 - c0)
  na = n0 + alpha * (n1 - n0)
  na = na / jnp.linalg.norm(na, axis=-1, keepdims=True)
  rel = pa - ca
  inplane = rel - jnp.sum(rel * na, axis=-1, keepdims=True) * na
  inside = jnp.linalg.norm(inplane, axis=-1) < radius + 0.5 * RADIUS
  face = side0[:, None] * na
  surf_v = vpad + jnp.cross(wpad, rel)
  u = v1 - surf_v
  approaching = jnp.sum(u * face, axis=-1) < 0
  hit = can_hit & inside & approaching & (crossed | close)
  ub, wb = _impact(u, w1, face, re, rmu)
  vb = ub + surf_v
  pb = ca + inplane + face * (RADIUS + 1e-3)
  h = hit[:, None]
  return jnp.where(h, pb, p1), jnp.where(h, vb, v1), jnp.where(h, wb, w1), hit, side0


# ---------------------------------------------------------------- human strokes
_ARRAYS = ("q", "blade", "offsets", "lengths", "impact", "cls", "imp_blade", "imp_vel", "imp_speed")
_STATIC = ("hz", "k", "backhand_from_y")


@jax.tree_util.register_pytree_node_class
class RefTables:
  """g1tt.reference.RefLib as device arrays (clips packed end to end); a
  pytree, so the tables are passed into compiled code instead of baked in."""

  @classmethod
  def from_lib(cls, lib, dtype=jnp.float32):
    t = cls.__new__(cls)
    t.hz, t.k, t.backhand_from_y = float(lib.hz), int(lib.k), float(lib.BACKHAND_FROM_Y)
    for f in _ARRAYS:
      x = getattr(lib, f)
      setattr(t, f, jnp.asarray(x, jnp.int32 if f in ("offsets", "lengths", "impact", "cls") else dtype))
    return t

  def tree_flatten(self):
    return tuple(getattr(self, f) for f in _ARRAYS), tuple(getattr(self, f) for f in _STATIC)

  @classmethod
  def tree_unflatten(cls, aux, children):
    t = cls.__new__(cls)
    for f, x in zip(_ARRAYS, children):
      setattr(t, f, x)
    for f, x in zip(_STATIC, aux):
      setattr(t, f, x)
    return t

  def choose(self, key, p_rel, v_star):
    """RefLib.choose(by_side=True): stroke type by contact side, one of the k
    nearest clips of that type at random, slowed to the commanded speed."""
    d = jnp.linalg.norm(p_rel[:, None, :] - self.imp_blade[None], axis=-1)
    backhand = p_rel[:, 1] > self.backhand_from_y
    wrong = jnp.where(backhand[:, None], self.cls[None] != 4, self.cls[None] != 1)
    d = jnp.where(wrong, jnp.inf, d)
    _, near = jax.lax.top_k(-d, self.k)
    pick = near[jnp.arange(p_rel.shape[0]), jax.random.randint(key, (p_rel.shape[0],), 0, self.k)]
    speed = jnp.maximum(jnp.linalg.norm(v_star, axis=-1), 0.5)
    stretch = jnp.clip(self.imp_speed[pick] / speed, 1.0, 2.5)
    return pick, stretch

  def window(self, clip, stretch):
    pre = self.impact[clip] / self.hz * stretch
    post = (self.lengths[clip] - 1 - self.impact[clip]) / self.hz * stretch
    return pre, post

  def _interp(self, table, clip, rel_t, stretch):
    f = self.impact[clip] + rel_t / stretch * self.hz
    f = jnp.clip(f, 0, self.lengths[clip] - 1)
    i0 = jnp.floor(f).astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, self.lengths[clip] - 1)
    a = (f - i0)[:, None]
    base = self.offsets[clip]
    return (1 - a) * table[base + i0] + a * table[base + i1]

  def blade_path(self, clip, rel_t, stretch, shift):
    return self._interp(self.blade, clip, rel_t, stretch) + shift

  def joints(self, clip, rel_t, stretch):
    return self._interp(self.q, clip, rel_t, stretch)


def np_float(x):
  return np.asarray(x, dtype=np.float32)
