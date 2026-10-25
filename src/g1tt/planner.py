"""HITTER-style model-based layer: a serving machine and the hit planner.

The planner predicts the incoming ball, picks where it crosses a virtual
hitting plane after bouncing on the robot's side, chooses a landing point on
the opponent's side, and inverts the rubber bounce model to get what the
paddle must do at impact: blade centre, face normal and velocity. The RL
whole-body controller is trained to deliver that command on time.
"""

from dataclasses import dataclass, field

import numpy as np

from g1tt import ball, ball_jit
from g1tt.ball import BallParams
from g1tt.model import NET_H, TABLE_H, TABLE_LEN

BALL_Z = TABLE_H + ball.RADIUS  # ball centre height when touching the table


@dataclass
class LaunchCfg:
  """Serving machine on the opponent's side. Ranges are (lo, hi)."""
  start_x: tuple = (1.5, 1.8)
  start_y: tuple = (-0.4, 0.4)
  start_z: tuple = (0.85, 1.05)
  # First bounce, on the robot's half.
  bounce_x: tuple = (-1.30, -0.95)
  bounce_y: tuple = (-0.5, 0.35)  # both sides: forehand (-y) and backhand (+y)
  flight_t: tuple = (0.42, 0.6)
  # Spin in rad/s. For a ball travelling -x, topspin is -w_y.
  topspin: tuple = (-250.0, 120.0)
  sidespin: tuple = (-80.0, 80.0)


@dataclass
class PlanCfg:
  # Virtual hitting plane, ball centre. Placed where retargeted human drives
  # (TTMD6) meet the ball on a G1: just behind the end line, 0.78-1.0 m up.
  hit_x: float = -1.40
  hit_z: tuple = (0.78, 1.00)
  hit_y: tuple = (-0.75, 0.45)
  # Landing target on the opponent's half and how long the return may fly.
  # Aim deep and central: returns land ~0.3 m from their aim, so a
  # target near the net or the side line is a likely fault.
  land_x: tuple = (0.65, 1.05)
  land_y: tuple = (-0.25, 0.25)
  return_t: tuple = field(default_factory=lambda: (0.40, 0.48, 0.56, 0.64, 0.72, 0.80))
  net_clearance: float = 0.10
  # Rally partner mode: prefer returns the other robot can play, i.e. that
  # cross its hitting plane (after bouncing on its half) inside this band
  # of its own frame.
  partner_band: bool = False
  partner_z: tuple = (0.80, 0.97)
  partner_y: tuple = (-0.75, 0.45)  # the trained reach band, partner's frame
  # Choose the return whose outgoing ball speed is closest to this (m/s)
  # instead of the one needing the slowest blade. The trained robot hits at
  # ~6.4 m/s whatever it is asked; planned soft returns (~5.4) went long.
  prefer_out_speed: float | None = None
  # Partner mode: resample the landing point this many times for returns that
  # would not reach the partner's band (wide aims rarely succeed at the first try).
  land_retries: int = 0
  horizon: float = 1.6


def _u(rng, rng_pair, n):
  return rng.uniform(rng_pair[0], rng_pair[1], n)


def sample_serves(rng: np.random.Generator, n: int, cfg: LaunchCfg, prm: BallParams, p0=None):
  """Initial (p, v, w) for n serves landing in cfg's bounce box.

  p0: where each ball is struck from; the serving machine's box if None, or
  the ball's own position when the opponent returns it in a rally.
  """
  if p0 is None:
    p0 = np.stack([_u(rng, cfg.start_x, n), _u(rng, cfg.start_y, n), _u(rng, cfg.start_z, n)], -1)
  tgt = np.stack([_u(rng, cfg.bounce_x, n), _u(rng, cfg.bounce_y, n), np.full(n, BALL_Z)], -1)
  w = np.stack([rng.uniform(-30, 30, n), _u(rng, cfg.topspin, n), _u(rng, cfg.sidespin, n)], -1)
  v0 = ball.solve_launch(p0, tgt, _u(rng, cfg.flight_t, n), w, prm)
  return p0, v0, w


@dataclass
class Plan:
  valid: np.ndarray  # (N,) bool
  t_hit: np.ndarray  # (N,) seconds from the planning instant
  ball_pos: np.ndarray  # (N, 3) ball centre at impact
  ball_vel_in: np.ndarray  # (N, 3)
  ball_vel_out: np.ndarray  # (N, 3) desired
  blade_pos: np.ndarray  # (N, 3) blade centre at impact
  normal: np.ndarray  # (N, 3) unit face normal, pointing at the ball
  blade_vel: np.ndarray  # (N, 3)
  land: np.ndarray  # (N, 3) intended landing point


def invert_paddle(v_in, w_in, v_out, e, mu, iters: int = 6):
  """Face normal and blade velocity that turn v_in into v_out.

  Frictionless, the impulse is along the normal, so n is the direction of
  v_out - v_in and the blade speed along n follows from restitution. Friction
  and spin bend the result; correct the aim by the miss and re-invert.
  """
  aim = v_out.copy()
  for _ in range(iters):
    d = aim - v_in
    n = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    vpn = (np.sum(aim * n, -1) + e * np.sum(v_in * n, -1)) / (1.0 + e)
    vp = vpn[:, None] * n
    u, _ = ball.impact(v_in - vp, w_in, n, e, mu)
    aim = aim + (v_out - (u + vp))
  return n, vp


def _solve_return(p_hit, w_out, land, prm: BallParams, cfg: PlanCfg, v_in=None, w_in=None):
  """Outgoing velocity to land at `land`, over a flight that clears the net.

  With the incoming ball (v_in, w_in) given, choose among the clearing flight
  times the one that needs the slowest blade: a G1 swings far slower than a
  player, and a slow, high return is one it can actually make. Without it,
  take the shortest clearing flight. All candidate times are solved as one
  batch.
  """
  n = p_hit.shape[0]
  ts = np.asarray(cfg.return_t, dtype=float)
  c = len(ts)
  rows = np.tile(np.arange(n), c)
  tt = np.repeat(ts, n)
  P, L, Wo = p_hit[rows], land[rows], w_out[rows]
  pr = prm.take(rows)
  v = ball.solve_launch(P, L, tt, Wo, pr, iters=3, h=0.01)
  # Height when crossing the net, from the planned flight.
  if ball_jit.ENABLED:
    z_net = ball_jit.net_height(P, v, Wo, pr, 0.01, int(ts.max() / 0.01) + 1)
  else:
    p, vv, ww = P.copy(), v.copy(), Wo.copy()
    z_net = np.full(n * c, -np.inf)
    for _ in range(int(ts.max() / 0.01) + 1):
      p_prev = p
      p, vv, ww = ball.flight(p, vv, ww, 0.01, pr.kd, pr.km)
      crossing = (p_prev[:, 0] < 0) & (p[:, 0] >= 0)
      f = -p_prev[:, 0] / np.where(crossing, p[:, 0] - p_prev[:, 0], 1.0)
      z_net = np.where(crossing, p_prev[:, 2] + f * (p[:, 2] - p_prev[:, 2]), z_net)
  ok = z_net - ball.RADIUS > TABLE_H + NET_H + cfg.net_clearance
  if v_in is None:
    cost = np.where(ok, tt, np.inf)  # shortest clearing time
  else:
    if cfg.prefer_out_speed is not None:
      cost = np.where(ok, np.abs(np.linalg.norm(v, axis=-1) - cfg.prefer_out_speed), np.inf)
    else:
      _, vp = invert_paddle(v_in[rows], w_in[rows], v, pr.rubber_e, pr.rubber_mu, iters=3)
      cost = np.where(ok, np.linalg.norm(vp, axis=-1), np.inf)
    if cfg.partner_band:
      cost = cost + np.where(_partner_ok(P, v, Wo, pr, cfg), 0.0, 100.0)
  cost = cost.reshape(c, n)
  best = cost.argmin(axis=0)
  pick = best * n + np.arange(n)
  return v[pick], np.isfinite(cost[best, np.arange(n)])


def _partner_ok(p, v, w, prm: BallParams, cfg: PlanCfg, horizon: float = 1.3):
  """Will this return reach the other robot inside its reachable band?

  The other robot's hitting plane is at x = -cfg.hit_x in our frame; in its
  frame heights are the same and y flips sign.
  """
  if ball_jit.ENABLED:
    has, pos = ball_jit.first_crossing(p, v, w, prm, -cfg.hit_x, 1, horizon, 0.006)
  else:
    P, V, _, B = ball.predict(p, v, w, prm, horizon=horizon, h=0.006)
    at = (B == 1) & (P[..., 0] >= -cfg.hit_x) & (V[..., 0] > 0)
    has = at.any(axis=0)
    k = np.where(has, at.argmax(axis=0), 0)
    pos = P[k, np.arange(p.shape[0])]
  y_theirs = -pos[:, 1]
  return (has & (pos[:, 2] > cfg.partner_z[0]) & (pos[:, 2] < cfg.partner_z[1])
          & (y_theirs > cfg.partner_y[0] + 0.05) & (y_theirs < cfg.partner_y[1] - 0.05))


def _return_for(bp, bv, bw, land, prm, cfg):
  """Outgoing velocity, face and blade velocity to return a ball at bp to `land`.

  The spin the hit puts on the ball bends the return, and that spin depends
  on the stroke: solve without it, invert, predict it, and solve again.
  """
  if ball_jit.ENABLED:
    return ball_jit.return_for(bp, bv, bw, land, prm, cfg)
  w_out = np.zeros_like(bp)
  for _ in range(2):
    v_out, ok = _solve_return(bp, w_out, land, prm, cfg, v_in=bv, w_in=bw)
    normal, blade_vel = invert_paddle(bv, bw, v_out, prm.rubber_e, prm.rubber_mu)
    _, w_out = ball.impact(bv - blade_vel, bw, normal, prm.rubber_e, prm.rubber_mu)
  return v_out, ok, normal, blade_vel


def plan_hits(p, v, w, prm: BallParams, cfg: PlanCfg, rng: np.random.Generator, land=None,
              bounces_left: int = 1, strict: bool = True, horizon: float | None = None) -> Plan:
  """Plan the return of each ball from its current state.

  bounces_left: table bounces still to come on the robot's side before the
  hit, 1 at the serve and 0 once the ball has bounced.
  strict: require the hit inside cfg's reachable band. Right for choosing
  serves; after the bounce the robot should chase the ball wherever it is.
  """
  n = p.shape[0]
  horizon = cfg.horizon if horizon is None else horizon
  P, V, W, B = ball.predict(p, v, w, prm, horizon)
  h = horizon / P.shape[0]
  # First sample after exactly one bounce where the ball reaches the hitting plane.
  at_plane = (B == bounces_left) & (P[..., 0] <= cfg.hit_x) & (V[..., 0] < 0)
  has = at_plane.any(axis=0)
  k = np.where(has, at_plane.argmax(axis=0), 0)
  idx = np.arange(n)
  bp, bv, bw = P[k, idx], V[k, idx], W[k, idx]
  t_hit = (k + 1) * h

  valid = has.copy()
  if strict:
    valid &= (bp[:, 2] > cfg.hit_z[0]) & (bp[:, 2] < cfg.hit_z[1])
    valid &= (bp[:, 1] > cfg.hit_y[0]) & (bp[:, 1] < cfg.hit_y[1])

  fixed_land = land is not None
  if land is None:
    land = np.stack([_u(rng, cfg.land_x, n), _u(rng, cfg.land_y, n), np.full(n, BALL_Z)], -1)
  v_out, ok, normal, blade_vel = _return_for(bp, bv, bw, land, prm, cfg)
  if cfg.partner_band and cfg.land_retries and not fixed_land:
    # Balls with no hit to make are not worth new landing spots.
    reach = (_partner_ok(bp, v_out, np.zeros((n, 3)), prm, cfg) & ok) | ~valid
    for _ in range(cfg.land_retries):
      redo = np.flatnonzero(~reach)
      if len(redo) == 0:
        break
      cand = land.copy()
      cand[redo, 0] = _u(rng, cfg.land_x, len(redo))
      cand[redo, 1] = _u(rng, cfg.land_y, len(redo))
      sub = prm.take(redo)
      v2, ok2, n2, b2 = _return_for(bp[redo], bv[redo], bw[redo], cand[redo], sub, cfg)
      good = ok2 & _partner_ok(bp[redo], v2, np.zeros((len(redo), 3)), sub, cfg)
      g = redo[good]
      land[g], v_out[g], ok[g], normal[g], blade_vel[g] = cand[g], v2[good], ok2[good], n2[good], b2[good]
      reach[g] = True
  valid &= ok
  blade_pos = bp - normal * ball.RADIUS
  return Plan(valid, t_hit, bp, bv, v_out, blade_pos, normal, blade_vel, land)


def serve_and_plan(rng, n, launch: LaunchCfg, plan: PlanCfg, prm_true: BallParams, prm_model: BallParams,
                   tries: int = 6, p0_fixed=None):
  """Serves whose plan (made with the planner's model, `prm_model`) is valid.

  The ball flies with `prm_true`; the planner only knows `prm_model`, which is
  how domain randomisation turns into plan error the controller must absorb.
  Returns (p0, v0, w0, plan, valid); invalid rows after `tries` stay invalid.
  """
  p0 = np.zeros((n, 3))
  v0 = np.zeros((n, 3))
  w0 = np.zeros((n, 3))
  todo = np.ones(n, dtype=bool)
  plan_out = None
  for _ in range(tries):
    m = int(todo.sum())
    if m == 0:
      break
    ids = np.flatnonzero(todo)
    pp, vv, ww = sample_serves(rng, m, launch, prm_true.take(ids), None if p0_fixed is None else p0_fixed[ids])
    pl = plan_hits(pp, vv, ww, prm_model.take(ids), plan, rng)
    if plan_out is None:
      plan_out = Plan(np.zeros(n, bool), np.zeros(n), *(np.zeros((n, 3)) for _ in range(7)))
    take = ids[pl.valid] if pl.valid.any() else ids[:0]
    # Last try: keep whatever we have so rows are filled.
    fill = ids if _ == tries - 1 else take
    sel = np.isin(ids, fill)
    p0[fill], v0[fill], w0[fill] = pp[sel], vv[sel], ww[sel]
    for f in ("valid", "t_hit", "ball_pos", "ball_vel_in", "ball_vel_out", "blade_pos", "normal", "blade_vel", "land"):
      getattr(plan_out, f)[fill] = getattr(pl, f)[sel]
    todo[take] = False
  return p0, v0, w0, plan_out, ~todo
