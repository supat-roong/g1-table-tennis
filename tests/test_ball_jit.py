import dataclasses

import numpy as np
import pytest

from g1tt import ball, ball_jit, planner
from g1tt.ball import BallParams

# float32 (the default) agrees with the float64 NumPy code to ~1e-5; float64 to ~1e-15.
TOL = 1e-9 if ball_jit.X64 else 2e-3


@pytest.fixture
def numpy_ball(monkeypatch):
  def use(flag):
    monkeypatch.setattr(ball_jit, "ENABLED", not flag)
  return use


def _balls(rng, n):
  prm = BallParams.randomised(rng, n)
  p = np.c_[rng.uniform(1.2, 1.5, n), rng.uniform(-0.5, 0.5, n), rng.uniform(0.85, 1.05, n)]
  v = np.c_[rng.uniform(-7, -4, n), rng.uniform(-0.5, 0.5, n), rng.uniform(0.5, 2.5, n)]
  w = rng.uniform(-150, 150, (n, 3))
  return p, v, w, prm


@pytest.mark.parametrize("n", [1, 5, 40])
def test_predict_and_solve_launch_match_numpy(numpy_ball, n):
  rng = np.random.default_rng(n)
  p, v, w, prm = _balls(rng, n)
  numpy_ball(True)
  ref = ball.predict(p, v, w, prm, 1.6, 0.004)
  tf = rng.choice([0.4, 0.56, 0.8], n)
  tgt = np.c_[rng.uniform(-1.1, -0.85, n), rng.uniform(-0.4, 0.4, n), np.full(n, planner.BALL_Z)]
  ref_v = ball.solve_launch(p, tgt, tf, w, prm, iters=3, h=0.01)
  numpy_ball(False)
  out = ball.predict(p, v, w, prm, 1.6, 0.004)
  for a, b in zip(out[:3], ref[:3]):
    np.testing.assert_allclose(a, b, atol=TOL)
  np.testing.assert_array_equal(out[3], ref[3])
  np.testing.assert_allclose(ball.solve_launch(p, tgt, tf, w, prm, iters=3, h=0.01), ref_v, atol=TOL)


def test_rally_plans_match_numpy(numpy_ball):
  """plan_hits in rally mode (partner band, landing retries) gives the same plans either way."""
  rng = np.random.default_rng(3)
  p, v, w, prm = _balls(rng, 24)
  cfg = dataclasses.replace(planner.PlanCfg(), partner_band=True, land_retries=4, prefer_out_speed=6.3,
                            land_y=(-0.45, 0.55), land_x=(0.85, 1.10))
  plans = []
  for numpy_on in (True, False):
    numpy_ball(numpy_on)
    plans.append(planner.plan_hits(p, v, w, prm, cfg, np.random.default_rng(7)))
  a, b = plans
  np.testing.assert_array_equal(a.valid, b.valid)
  for f in ("t_hit", "ball_pos", "ball_vel_out", "blade_pos", "normal", "blade_vel", "land"):
    np.testing.assert_allclose(getattr(a, f), getattr(b, f), atol=TOL, err_msg=f)


@pytest.mark.parametrize("prefer, band", [(6.3, True), (None, True), (None, False)])
def test_return_for_matches_numpy(numpy_ball, prefer, band):
  rng = np.random.default_rng(11)
  n = 17
  bp = np.c_[np.full(n, -1.4), rng.uniform(-0.6, 0.4, n), rng.uniform(0.8, 1.0, n)]
  bv = np.c_[rng.uniform(-6, -3, n), rng.uniform(-0.5, 0.5, n), rng.uniform(-1.5, 1.0, n)]
  bw = rng.uniform(-150, 150, (n, 3))
  land = np.c_[rng.uniform(0.85, 1.1, n), rng.uniform(-0.45, 0.55, n), np.full(n, planner.BALL_Z)]
  prm = BallParams.randomised(rng, n)
  cfg = dataclasses.replace(planner.PlanCfg(), partner_band=band, prefer_out_speed=prefer)
  numpy_ball(True)
  ref = planner._return_for(bp, bv, bw, land, prm, cfg)
  numpy_ball(False)
  out = planner._return_for(bp, bv, bw, land, prm, cfg)
  np.testing.assert_array_equal(out[1], ref[1])
  for a, b in zip((out[0], out[2], out[3]), (ref[0], ref[2], ref[3])):
    np.testing.assert_allclose(a, b, atol=TOL)


def test_return_shortfall_matches_numpy(numpy_ball):
  rng = np.random.default_rng(5)
  n = 30
  p = np.c_[np.full(n, -1.4), rng.uniform(-0.5, 0.5, n), rng.uniform(0.8, 1.0, n)]
  v = np.c_[rng.uniform(2, 8, n), rng.uniform(-1, 1, n), rng.uniform(-1, 3, n)]
  w = rng.uniform(-150, 150, (n, 3))
  prm = BallParams.randomised(rng, n)
  numpy_ball(True)
  ref = ball.return_shortfall(p, v, w, prm)
  numpy_ball(False)
  np.testing.assert_allclose(ball.return_shortfall(p, v, w, prm), ref, atol=TOL)
