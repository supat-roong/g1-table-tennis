import numpy as np

from g1tt import ball, planner
from g1tt.ball import BallParams
from g1tt.model import TABLE_LEN


def test_invert_paddle_reproduces_outgoing_velocity():
  rng = np.random.default_rng(0)
  n = 64
  v_in = np.stack([rng.uniform(4, 7, n), rng.uniform(-1, 1, n), rng.uniform(-1, 2, n)], -1) * [-1, 1, 1]
  w_in = rng.uniform(-200, 200, (n, 3))
  v_out = np.stack([rng.uniform(3, 6, n), rng.uniform(-1, 1, n), rng.uniform(0.5, 2.5, n)], -1)
  prm = BallParams.nominal(n)
  normal, vp = planner.invert_paddle(v_in, w_in, v_out, prm.rubber_e, prm.rubber_mu)
  u, _ = ball.impact(v_in - vp, w_in, normal, prm.rubber_e, prm.rubber_mu)
  err = np.linalg.norm(u + vp - v_out, axis=-1)
  assert np.percentile(err, 90) < 0.05
  assert np.allclose(np.linalg.norm(normal, axis=-1), 1.0)


def test_planned_hit_lands_on_target_with_perfect_execution():
  """Serve, plan, apply the planned blade motion at the planned instant, and
  check the return lands near the planned point on the opponent's half."""
  rng = np.random.default_rng(1)
  n = 128
  prm = BallParams.nominal(n)
  p0, v0, w0, plan, ok = planner.serve_and_plan(rng, n, planner.LaunchCfg(), planner.PlanCfg(), prm, prm)
  assert ok.mean() > 0.9
  ids = np.flatnonzero(ok)
  p, v, w = p0[ids], v0[ids], w0[ids]
  sub = prm.take(ids)
  h = 0.002
  steps = np.round(plan.t_hit[ids] / h).astype(int)
  alive = np.ones(len(ids), bool)
  hit_done = np.zeros(len(ids), bool)
  landed = np.full((len(ids), 3), np.nan)
  for k in range(1, 1200):
    p, v, w, table, net, floor = ball.step_world(p, v, w, h, sub, alive)
    now = steps == k
    if now.any():
      u, w2 = ball.impact(v[now] - plan.blade_vel[ids][now], w[now], plan.normal[ids][now],
                          sub.rubber_e[now], sub.rubber_mu[now])
      v[now] = u + plan.blade_vel[ids][now]
      w[now] = w2
      hit_done |= now
    first = table & hit_done & np.isnan(landed[:, 0])
    landed[first] = p[first]
    alive &= ~(net | floor)
  assert hit_done.all()
  on_opp = (landed[:, 0] > 0) & (landed[:, 0] < TABLE_LEN / 2)
  assert on_opp.mean() > 0.9
  # Spin after the hit is ignored by the return solve, so allow some slack.
  miss = np.linalg.norm(landed[on_opp, :2] - plan.land[ids][on_opp, :2], axis=-1)
  assert np.median(miss) < 0.15


def test_replan_after_the_bounce_is_valid_and_close_to_the_serve_plan():
  """Re-planning from a ball that has already bounced must find the hit."""
  rng = np.random.default_rng(2)
  n = 64
  prm = BallParams.nominal(n)
  p0, v0, w0, plan, ok = planner.serve_and_plan(rng, n, planner.LaunchCfg(), planner.PlanCfg(), prm, prm)
  p, v, w = p0.copy(), v0.copy(), w0.copy()
  alive = np.ones(n, bool)
  bounced = np.zeros(n, bool)
  t = np.zeros(n)
  state = [None] * n
  for _ in range(800):
    p, v, w, table, net, floor = ball.step_world(p, v, w, 0.001, prm, alive)
    t += 0.001 * ~bounced
    first = table & ~bounced
    for i in np.flatnonzero(first):
      state[i] = (p[i].copy(), v[i].copy(), w[i].copy(), t[i])
    bounced |= table
  ids = np.array([i for i in range(n) if ok[i] and state[i] is not None])
  P = np.stack([state[i][0] for i in ids])
  V = np.stack([state[i][1] for i in ids])
  W = np.stack([state[i][2] for i in ids])
  re = planner.plan_hits(P, V, W, prm.take(ids), planner.PlanCfg(), rng, land=plan.land[ids], bounces_left=0)
  assert re.valid.mean() > 0.95
  t_bounce = np.array([state[i][3] for i in ids])
  # Same ball, same model: the re-plan agrees with the serve plan.
  assert np.median(np.abs(t_bounce + re.t_hit - plan.t_hit[ids])) < 0.01
  assert np.median(np.linalg.norm(re.ball_pos - plan.ball_pos[ids], axis=-1)) < 0.02
