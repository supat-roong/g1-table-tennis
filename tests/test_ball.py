import numpy as np

from g1tt import ball
from g1tt.model import TABLE_H


def _drop(height, w=None, vx=0.0):
  prm = ball.BallParams.nominal(1)
  p = np.array([[0.5, 0.0, TABLE_H + ball.RADIUS + height]])
  v = np.array([[vx, 0.0, 0.0]])
  w = np.zeros((1, 3)) if w is None else np.array([w], dtype=float)
  alive = np.ones(1, bool)
  apex, bounced, v_after = 0.0, False, None
  for _ in range(2000):
    p, v, w, table, net, floor = ball.step_world(p, v, w, 0.001, prm, alive)
    if table.any():
      bounced, v_after = True, v.copy()
    if bounced:
      apex = max(apex, p[0, 2] - TABLE_H - ball.RADIUS)
      if v[0, 2] < 0:
        break
  return apex, v_after


def test_drop_rebound_matches_itTF():
  apex, _ = _drop(0.30)
  # ITTF: dropped from 30 cm a legal ball rebounds about 23 cm.
  assert 0.21 < apex < 0.25


def test_topspin_kicks_forward_backspin_checks():
  # Ball travelling +x; topspin is +w_y (top surface moving forward).
  # At 1 m/s a flat ball stops slipping mid-contact; heavy backspin keeps sliding and checks.
  u = np.array([[1.0, 0, -2.5]] * 3)
  w = np.array([[0, 0, 0], [0, 300, 0], [0, -300, 0]])
  n = np.array([[0, 0, 1.0]] * 3)
  e = np.full(3, ball.TABLE_E)
  mu = np.full(3, ball.TABLE_MU)
  u2, w2 = ball.impact(u, w, n, e, mu)
  flat, top, back = u2[:, 0]
  assert top > flat > back
  assert np.allclose(u2[:, 2], 2.5 * ball.TABLE_E)
  # Friction converts some slip into spin: a non-spinning ball leaves with topspin.
  assert w2[0, 1] > 0


def test_magnus_topspin_dips():
  prm = ball.BallParams.nominal(2)
  p = np.zeros((2, 3)) + [0, 0, 2.0]
  v = np.array([[8.0, 0, 0], [8.0, 0, 0]])
  w = np.array([[0, 0, 0], [0, 400.0, 0]])
  for _ in range(300):
    p, v, w = ball.flight(p, v, w, 0.001, prm.kd, prm.km)
  assert p[1, 2] < p[0, 2] - 0.05


def test_fast_ball_cannot_tunnel_through_blade():
  prm = ball.BallParams.nominal(1)
  n = np.array([[-1.0, 0, 0]])
  c = np.array([[0.0, 0, 1.0]])
  p0 = np.array([[0.015, 0, 1.0]])  # just in front of the blade, closing at 30 m/s
  v = np.array([[-30.0, 0, 0]])
  p1 = p0 + v * 0.001  # ends 1.5 cm behind the plane: a naive overlap test misses it
  p, v2, w2, hit, side = ball.paddle_contact(
    p0, p1, v, np.zeros((1, 3)), c, c, n, n, np.zeros((1, 3)), np.zeros((1, 3)), 0.076, np.ones(1, bool), prm)
  assert hit[0]
  assert v2[0, 0] > 0  # sent back
  assert p[0, 0] > 0  # and put back on the ball's side


def test_moving_paddle_adds_speed():
  prm = ball.BallParams.nominal(2)
  n = np.array([[1.0, 0, 0]] * 2)
  c = np.zeros((2, 3)) + [0, 0, 1.0]
  p0 = np.array([[0.03, 0, 1.0]] * 2)
  v = np.array([[-5.0, 0, 0]] * 2)
  p1 = p0 + v * 0.002
  vpad = np.array([[0.0, 0, 0], [3.0, 0, 0]])
  _, vout, _, hit, _ = ball.paddle_contact(
    p0, p1, v, np.zeros((2, 3)), c, c, n, n, vpad, np.zeros((2, 3)), 0.076, np.ones(2, bool), prm)
  assert hit.all()
  assert vout[1, 0] > vout[0, 0] + 4.0  # (1 + e) * 3 m/s more


def test_miss_outside_blade_disc():
  prm = ball.BallParams.nominal(1)
  n = np.array([[-1.0, 0, 0]])
  c = np.array([[0.0, 0, 1.0]])
  p0 = np.array([[0.015, 0.2, 1.0]])
  v = np.array([[-30.0, 0, 0]])
  _, _, _, hit, _ = ball.paddle_contact(
    p0, p0 + v * 0.001, v, np.zeros((1, 3)), c, c, n, n, np.zeros((1, 3)), np.zeros((1, 3)), 0.076,
    np.ones(1, bool), prm)
  assert not hit[0]


def test_solve_launch_hits_target():
  prm = ball.BallParams.nominal(3)
  p0 = np.array([[1.6, 0.0, 1.0], [1.7, 0.3, 0.9], [1.5, -0.3, 1.1]])
  tgt = np.array([[-0.8, -0.2, TABLE_H + ball.RADIUS], [-0.6, 0.0, TABLE_H + ball.RADIUS],
                  [-1.0, -0.4, TABLE_H + ball.RADIUS]])
  T = np.array([0.45, 0.5, 0.4])
  w = np.array([[0, -200.0, 0], [0, 100, 0], [0, 0, 80]])
  v0 = ball.solve_launch(p0, tgt, T, w, prm)
  p, v, ww = p0.copy(), v0.copy(), w.copy()
  for k in range(int(round(0.5 / 0.002))):
    p, v, ww = ball.flight(p, v, ww, 0.002, prm.kd, prm.km)
    if k + 1 == 225:
      assert np.linalg.norm(p[0] - tgt[0]) < 0.01
  assert np.linalg.norm(p[1] - tgt[1]) < 0.01


def test_return_shortfall_grades_net_and_long_balls():
  prm = ball.BallParams.nominal(3)
  p = np.array([[-1.4, 0.0, 1.0]] * 3)
  w = np.zeros((3, 3))
  good = ball.solve_launch(p[:1], np.array([[0.8, 0.0, TABLE_H + ball.RADIUS]]), np.array([0.6]), w[:1],
                           prm.take([0]))
  # Lands; flat and slow (comes down on our side ~0.6 m short of the net); far long.
  v = np.array([good[0], [4.0, 0.0, 0.0], [12.0, 0.0, 2.0]])
  sf = ball.return_shortfall(p, v, w, prm)
  assert sf[0] == 0.0
  assert 0.4 < sf[1] < 0.8
  assert sf[2] > 1.0
