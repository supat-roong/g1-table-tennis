"""Human reference strokes for the controller, from the retargeted TTMD6 library.

When the planner commits to a hit, the environment picks a human stroke whose
blade position at impact is near the planned one, slows it down if the human
swung faster than the command asks, and times it so the human's contact
frame lands on the planned impact instant. The policy is rewarded for
following that stroke's joint angles: HITTER's motion-reference idea, with
TTMD6 as the motion source.
"""

from pathlib import Path

import numpy as np

from g1tt.mocap import OUT as LIBRARY

UPPER = np.arange(12, 29)  # waist + both arms, indices into the 29 actuated joints
# Joints rewarded for following the human: waist, shoulders and elbows. TTMD6
# has no hand markers, so the retargeted wrist angles are an IK by-product,
# and imitating them locks the paddle face at the wrong angle.
IMITATE = np.array([12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25])
LEGS = np.arange(0, 12)


class RefLib:
  def __init__(self, path: Path = LIBRARY, classes=(1, 4), k_nearest: int = 8):
    z = np.load(path)
    keep = np.isin(z["cls"], classes)
    self.hz = float(z["hz"])
    idx = np.flatnonzero(keep)
    q = z["q"]
    blade = z["blade"]
    self.lengths = z["lengths"][idx]
    self.impact = z["impact"][idx]
    self.cls = z["cls"][idx]
    starts = z["offsets"][idx]
    # Repack the kept clips contiguously: joint angles only (drop the free joint).
    self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
    self.q = np.concatenate([q[s:s + n, 7:] for s, n in zip(starts, self.lengths)]).astype(np.float64)
    imp_abs = starts + self.impact
    # Blade at impact relative to where the robot stood at the clip's start,
    # so the library does not depend on where the robot is placed.
    home = q[starts, :3].astype(np.float64)
    home[:, 2] = 0.0
    self.imp_blade = blade[imp_abs].astype(np.float64) - home
    self.blade = np.concatenate([blade[s:s + n] - h for s, n, h in zip(starts, self.lengths, home)]).astype(np.float64)
    self.imp_vel = ((blade[imp_abs + 1] - blade[imp_abs - 1]) * self.hz / 2).astype(np.float64)
    self.imp_speed = np.linalg.norm(self.imp_vel, axis=-1)
    self.k = k_nearest

  def __len__(self):
    return len(self.lengths)

  # Contact points left of this (robot's frame, from its spot) get a
  # backhand, right of it a forehand, like a right-handed player.
  BACKHAND_FROM_Y = -0.15

  def choose(self, p_star, v_star, rng, by_side: bool = False):
    """Clip index and time stretch (>= 1, slower) for each planned hit.

    p_star is the planned blade centre relative to the robot's home spot
    (x, y from the spot, z from the floor). With by_side, the stroke type is
    set by which side of the body the contact is on, then the nearest clip of
    that type is taken.
    """
    d = np.linalg.norm(p_star[:, None, :] - self.imp_blade[None], axis=-1)
    if by_side:
      backhand = p_star[:, 1] > self.BACKHAND_FROM_Y
      wrong = np.where(backhand[:, None], self.cls[None] != 4, self.cls[None] != 1)
      d = np.where(wrong, np.inf, d)
    near = np.argpartition(d, self.k, axis=1)[:, : self.k]
    pick = near[np.arange(len(p_star)), rng.integers(0, self.k, len(p_star))]
    speed = np.maximum(np.linalg.norm(v_star, axis=-1), 0.5)
    stretch = np.clip(self.imp_speed[pick] / speed, 1.0, 2.5)
    return pick, stretch

  def window(self, clip, stretch):
    """Seconds of the stroke before and after impact, after stretching."""
    pre = self.impact[clip] / self.hz * stretch
    post = (self.lengths[clip] - 1 - self.impact[clip]) / self.hz * stretch
    return pre, post

  def _interp(self, table, clip, rel_t, stretch):
    f = self.impact[clip] + rel_t / stretch * self.hz
    f = np.clip(f, 0, self.lengths[clip] - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, self.lengths[clip] - 1)
    a = (f - i0)[:, None]
    base = self.offsets[clip]
    return (1 - a) * table[base + i0] + a * table[base + i1]

  def blade_path(self, clip, rel_t, stretch, shift):
    """Human paddle path, relative to the robot's home spot, shifted so the
    contact point lands on the planned blade position (shift = target - clip impact)."""
    return self._interp(self.blade, clip, rel_t, stretch) + shift

  def joints(self, clip, rel_t, stretch):
    """Reference joint angles (N, 29) at rel_t seconds from impact (negative = before)."""
    return self._interp(self.q, clip, rel_t, stretch)
