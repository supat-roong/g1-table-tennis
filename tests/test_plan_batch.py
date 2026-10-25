import dataclasses

import jax
import numpy as np

from g1tt import ball_jit, planner
from g1tt.ball import BallParams
from g1tt.plan_batch import plan_batch
from g1tt.reference import RefLib

TOL = 1e-6 if ball_jit.X64 else 5e-3


def _incoming(rng, n):
  """Balls just hit by the partner, coming at the robot (its frame)."""
  prm = BallParams.randomised(rng, n)
  p = np.c_[rng.uniform(1.3, 1.5, n), rng.uniform(-0.5, 0.5, n), rng.uniform(0.85, 1.05, n)]
  tgt = np.c_[rng.uniform(-1.2, -0.9, n), rng.uniform(-0.5, 0.35, n), np.full(n, planner.BALL_Z)]
  w = np.c_[rng.uniform(-30, 30, n), rng.uniform(-250, 120, n), rng.uniform(-80, 80, n)]
  v = planner.ball.solve_launch(p, tgt, rng.uniform(0.42, 0.6, n), w, prm)
  return p, v, w, prm


def _cfg():
  return dataclasses.replace(planner.PlanCfg(), partner_band=True, prefer_out_speed=6.3, land_x=(0.85, 1.10),
                             land_y=(0.10, 0.55), partner_y=(-0.75, -0.25), hit_y=(-1.15, 0.90), hit_z=(0.70, 1.20))


def _ranges(cfg, n):
  return dict(land_x=np.tile(cfg.land_x, (n, 1)), land_y=np.tile(cfg.land_y, (n, 1)),
              partner_y=np.tile(cfg.partner_y, (n, 1)), retries=np.zeros(n, int), clear=np.full(n, cfg.net_clearance))


def _compare(a, b):
  same = a.valid == b.valid
  assert same.mean() > 0.98, same.mean()  # float32 may flip a ball sitting on a band edge
  both = a.valid & b.valid
  for f in ("t_hit", "ball_pos", "ball_vel_out", "blade_pos", "normal", "blade_vel", "land"):
    np.testing.assert_allclose(getattr(a, f)[both], getattr(b, f)[both], atol=TOL, err_msg=f)


def test_fixed_landing_matches_plan_hits():
  rng = np.random.default_rng(0)
  n = 60
  p, v, w, prm = _incoming(rng, n)
  cfg = _cfg()
  land = np.c_[rng.uniform(*cfg.land_x, n), rng.uniform(*cfg.land_y, n), np.full(n, planner.BALL_Z)]
  ref = planner.plan_hits(p, v, w, prm, cfg, rng, land=land)
  out = plan_batch(p, v, w, prm, jax.random.PRNGKey(0), cfg=cfg, land=land, **_ranges(cfg, n))
  assert out.valid.mean() > 0.3
  _compare(out, ref)


def test_backhand_replan_matches_duel_rule():
  """Valid backhands re-planned with the backhand clearance, as duel._plan_for does."""
  rng = np.random.default_rng(1)
  n = 80
  p, v, w, prm = _incoming(rng, n)
  cfg = _cfg()
  land = np.c_[rng.uniform(*cfg.land_x, n), rng.uniform(*cfg.land_y, n), np.full(n, planner.BALL_Z)]
  ref = planner.plan_hits(p, v, w, prm, cfg, rng, land=land)
  hi = planner.plan_hits(p, v, w, prm, dataclasses.replace(cfg, net_clearance=0.30), rng, land=land)
  use = (ref.valid & (ref.blade_pos[:, 1] > RefLib.BACKHAND_FROM_Y) & hi.valid
         & (hi.blade_pos[:, 1] > RefLib.BACKHAND_FROM_Y))
  assert use.sum() >= 5
  for f in planner.Plan.__dataclass_fields__:
    getattr(ref, f)[use] = getattr(hi, f)[use]
  out = plan_batch(p, v, w, prm, jax.random.PRNGKey(0), cfg=cfg, land=land, bh_clear=0.30, **_ranges(cfg, n))
  _compare(out, ref)


def test_retries_raise_partner_reach_and_respect_counts():
  rng = np.random.default_rng(2)
  n = 120
  p, v, w, prm = _incoming(rng, n)
  cfg = _cfg()
  r = _ranges(cfg, n)
  none = plan_batch(p, v, w, prm, jax.random.PRNGKey(3), cfg=cfg, **r)
  r["retries"] = np.full(n, 4)
  four = plan_batch(p, v, w, prm, jax.random.PRNGKey(3), cfg=cfg, **r)
  reach = lambda pl: planner._partner_ok(pl.ball_pos, pl.ball_vel_out, np.zeros((n, 3)), prm, cfg)[pl.valid].mean()
  assert reach(four) >= reach(none)
  # Same key, no retries allowed: the first draw, identical to max_retries=0.
  r["retries"] = np.zeros(n, int)
  again = plan_batch(p, v, w, prm, jax.random.PRNGKey(3), cfg=cfg, max_retries=4, **r)
  np.testing.assert_allclose(again.land, none.land, atol=TOL)
