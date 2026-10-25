"""Two G1s rallying with the same policy, one at each end of the table.

    .venv/bin/python -m g1tt.duel runs/g1/ckpt.pt --matches 8 --seconds 90
    .venv/bin/python -m g1tt.duel runs/g1/ckpt.pt --video renders/match.mp4 --seconds 60
    .venv/bin/python -m g1tt.duel runs/g1/ckpt.pt --find-rally 16 --seconds 300

Robot B stands behind the +x end, turned half a circle. Everything B's policy
and planner see is expressed in a frame turned 180 degrees about z, where B
stands exactly where robot A stands: positions and velocities (x, y) ->
(-x, -y), spins likewise, yaw + pi. Joint angles, gyro and gravity are already
in the body frame. So one network drives both robots.

A serving machine starts each rally, alternating which robot receives;
neither robot serves. After that the robots return each other's balls: a
return must bounce on the other half, a robot may hit only once the ball has
bounced on its side, and the rally ends on a fault, a miss or a fall. Each
return is aimed at the partner's forehand, its backhand, or wide of its reach.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

from g1tt import ball, planner
from g1tt.ball import BallParams
from g1tt.env import ACTOR_DIM, BALL_TICKS, CRITIC_DIM, CTRL_DT, REF_LOOKAHEAD, SUBSTEPS, _action_scale
from g1tt.model import (BLADE_RADIUS, OPP_PREFIX, ROBOT_X, build_duel_model, build_model, default_joint_targets,
                        ready_qpos)
from g1tt.ppo import PPO, PPOCfg
from g1tt.reference import UPPER, RefLib
from g1tt.sgs import SuccessGuidedSampler

FLIP = np.array([-1.0, -1.0, 1.0])  # world <-> robot B's frame (180 degrees about z)


def _yaw(quat):
  w, x, y, z = quat.T
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_rot(yaw):
  c, s = np.cos(yaw), np.sin(yaw)
  R = np.zeros((len(yaw), 3, 3))
  R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, s, -s, c, 1.0
  return R


def _rot(R, x):
  return np.einsum("nij,nj->ni", R, x)


def load_policy(path, n):
  """The actor-critic of a training checkpoint, for n robots, in eval mode."""
  sd = torch.load(path, weights_only=False)
  cfg = PPOCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in sd["ppo"]["cfg"].items()})
  algo = PPO(ACTOR_DIM, CRITIC_DIM, 29, n, cfg)
  algo.load_state_dict(sd["ppo"], load_opt=False)
  algo.ac.eval()
  return algo.ac


class Side:
  """One robot's plan, reference stroke and last action, in its own frame."""

  def __init__(self, n, ref: RefLib, q_default, ready_blade, ready_normal, rng):
    self.n, self.ref, self.rng = n, ref, rng
    self.q_default, self.ready_blade, self.ready_normal = q_default, ready_blade, ready_normal
    self.plan_valid = np.zeros(n, bool)
    self.t_hit = np.full(n, np.inf)
    self.p_star = np.zeros((n, 3))
    self.n_star = np.zeros((n, 3))
    self.v_star = np.zeros((n, 3))
    self.hit_done = np.zeros(n, bool)
    self.ref_on = np.zeros(n, bool)
    self.ref_clip = np.zeros(n, dtype=np.int64)
    self.ref_stretch = np.ones(n)
    self.ref_timp = np.full(n, np.inf)
    self.ref_shift = np.zeros((n, 3))
    self.last_action = np.zeros((n, 29))
    self.prev_action = np.zeros((n, 29))
    self.vout_star = np.zeros((n, 3))
    self.land_tgt = np.zeros((n, 3))
    self.face = np.ones(n)  # +1 forehand (red, blade +y) to the ball, -1 backhand (black)
    self.wide = np.zeros(n, bool)  # this ball needs footwork
    self.aim = np.zeros(n, dtype=np.int64)  # where my return goes: 0 forehand, 1 wide, 3 backhand
    self.cell = np.full(n, -1, dtype=np.int64)  # with SGS: the return configuration (aim, depth, width)

  def clear(self, ids):
    self.plan_valid[ids] = False
    self.t_hit[ids] = np.inf
    self.hit_done[ids] = False
    self.ref_on[ids] = False
    self.ref_timp[ids] = np.inf

  def set_plan(self, ids, pl, t_now, keep_stroke=False):
    v = pl.valid
    ids_v = ids[v]
    self.plan_valid[ids] = v
    self.t_hit[ids_v] = t_now[v] + pl.t_hit[v]
    self.p_star[ids_v] = pl.blade_pos[v]
    self.v_star[ids_v] = pl.blade_vel[v]
    self.vout_star[ids_v] = pl.ball_vel_out[v]
    self.land_tgt[ids_v] = pl.land[v]
    new = ids_v if not keep_stroke else ids_v[~self.ref_on[ids_v]]
    if len(new):
      clip, stretch = self.ref.choose(self.p_star[new] - [ROBOT_X, 0.0, 0.0], self.v_star[new], self.rng,
                                      by_side=True)
      self.ref_clip[new], self.ref_stretch[new] = clip, stretch
      self.ref_on[new] = True
      self.face[new] = np.where(self.ref.cls[clip] == 4, -1.0, 1.0)
    # The commanded face normal is the blade's +y axis: the planner's normal
    # for a forehand, its opposite for a backhand (black face to the ball).
    self.n_star[ids_v] = pl.normal[v] * self.face[ids_v, None]
    self.ref_timp[ids_v] = self.t_hit[ids_v]
    on = ids_v[self.ref_on[ids_v]]
    self.ref_shift[on] = self.p_star[on] - [ROBOT_X, 0.0, 0.0] - self.ref.imp_blade[self.ref_clip[on]]

  def _window(self, t, ahead):
    rel = t + ahead - self.ref_timp
    pre, post = self.ref.window(self.ref_clip, self.ref_stretch)
    active = self.ref_on & (rel >= -pre) & (rel <= post)
    self.ref_on &= ~(rel > post)
    return active, rel

  def obs(self, t, base, yaw, gyro, grav, qj, qdj, blade):
    R = _yaw_rot(yaw)
    tau = self.t_hit - t
    act = self.plan_valid & ~self.hit_done & (tau > -0.06)
    p = np.where(act[:, None], self.p_star, base + self.ready_blade)
    nrm = np.where(act[:, None], self.n_star, self.ready_normal)
    vel = np.where(act[:, None], self.v_star, 0.0)
    tau_c = np.where(act, np.clip(tau, -0.1, 1.0), 1.0)
    command = np.concatenate([_rot(R, p - base), _rot(R, nrm), 0.3 * _rot(R, vel), tau_c[:, None], act[:, None]], -1)

    r_act, rel = self._window(t, REF_LOOKAHEAD)
    q = self.ref.joints(self.ref_clip, np.where(r_act, rel, 0.0), self.ref_stretch)
    q = np.where(r_act[:, None], q, self.q_default)
    ref_obs = np.concatenate([(q - self.q_default)[:, UPPER], r_act[:, None]], -1)

    off = np.zeros((self.n, 3))
    off[:, 0], off[:, 1] = base[:, 0] - ROBOT_X, base[:, 1]
    home = _rot(R, off)[:, :2]

    p_act, rel = self._window(t, REF_LOOKAHEAD)
    r = np.where(p_act, rel, 0.0)
    pt = self.ref.blade_path(self.ref_clip, r, self.ref_stretch, self.ref_shift)
    v_h = self.ref.imp_vel[self.ref_clip] / self.ref_stretch[:, None]
    pt = pt + (self.v_star - v_h) * (r * np.exp(-(r / 0.15) ** 2))[:, None] + [ROBOT_X, 0.0, 0.0]
    path = _rot(R, np.where(p_act[:, None], pt - base, 0.0))

    return np.concatenate([0.25 * gyro, grav, qj - self.q_default, 0.05 * qdj, self.last_action,
                           command, ref_obs, home, path], -1).astype(np.float32)


class Duel:
  # Balls a robot goes for: the band it reaches from its spot (PlanCfg.hit_y,
  # hit_z) widened by a step or two sideways and a little higher and lower.
  CHASE_Y = (-1.15, 0.90)
  CHASE_Z = (0.70, 1.20)
  # Deliberately wide returns, to make the partner move: where they cross its
  # hitting plane, in its frame, ~0.05-0.40 m beyond its standing reach, so a
  # step reaches them. Wider ones broke nearly every rally.
  WIDE_Y = ((-1.05, -0.80), (0.55, 0.85))
  # Landing spots that can make such a ball (short and near the side line);
  # deeper or more central ones cannot curve that far out (measured).
  WIDE_LAND_Y = ((0.40, 0.74), (-0.74, -0.40))
  # Returns are aimed at one stroke: a share where the partner meets the ball
  # on its backhand side, the rest on its forehand side, and wide ones only out
  # to its forehand. Backhands come off lower and slower than planned, so
  # backhand returns are planned 0.30 m over the net and deep (1.00-1.20 m);
  # with the usual clearance and depth half of them could not reach the
  # partner's hitting plane at all.
  BH_CLEAR = 0.30  # m over the net that backhand returns are planned to clear
  BH_LAND_X = (1.00, 1.20)  # landing depth for backhand returns
  FH_Y = (-0.75, -0.25)
  BH_Y = (-0.05, 0.40)
  FH_LAND_Y = (0.10, 0.55)
  BH_LAND_Y = (-0.45, -0.05)
  AIM_RETRIES = 4  # new landing spots tried for a return that misses the stroke's band
  REACH_HORIZON = 2.0  # s of flight checked for a playable return; lobbed backhands take longer than 1.3
  SOLID_AFTER = 2  # returns that must land after a shot for it to count as solid
  MOVED = 0.20  # pelvis this far from the spot at contact counts as moving to the ball
  WRIST = np.array([26, 27, 28])  # right wrist roll, pitch, yaw among the 29 actuated joints
  SGS_BINS = 3  # SGS return configurations: aim x this many landing depths x this many widths

  @staticmethod
  def base_plan_cfg() -> planner.PlanCfg:
    """The duel's planner settings (also used by g1tt.gpu without building a model)."""
    return planner.PlanCfg(land_x=(0.85, 1.10), land_y=(-0.45, 0.55), partner_band=True, partner_z=(0.78, 1.02),
                           prefer_out_speed=6.3)

  def __init__(self, n: int, seed: int = 0, dr_scale: float = 1.0, episode_s: float = 300.0,
               wide_frac: float = 0.10, bh_frac: float = 0.25, sgs: dict | None = None):
    self.n = n
    self.wide_frac, self.bh_frac = wide_frac, bh_frac
    # Success Guided Sampling (g1tt.sgs) of serves and return configurations,
    # for training: sgs = {target, kappa, temperature, history, mix}, mix being
    # the share of serves and returns it picks (the rest follow the usual mix).
    self.sgs = sgs
    self.rng = np.random.default_rng(seed)
    self.model = build_duel_model()
    m = self.model
    self.datas = [mujoco.MjData(m) for _ in range(n)]
    single = build_model()
    self.q0 = ready_qpos(single)
    self.q_default = default_joint_targets()
    self.act_scale = _action_scale()
    self.max_steps = int(round(episode_s / CTRL_DT))
    self.phys_dt = m.opt.timestep
    self.jnt_lo, self.jnt_hi = single.jnt_range[1:, 0], single.jnt_range[1:, 1]
    self.pelvis = [m.body("pelvis").id, m.body(OPP_PREFIX + "pelvis").id]
    self.blade_sid = [m.site("blade").id, m.site(OPP_PREFIX + "blade").id]
    self.ball_mocap = m.body("ball").mocapid[0]
    self.tgt_mocap = [m.body("hit_target").mocapid[0], m.body("hit_target_b").mocapid[0]]

    d = mujoco.MjData(single)
    d.qpos[:] = self.q0
    mujoco.mj_forward(single, d)
    sid = single.site("blade").id
    ready_blade = d.site_xpos[sid] - self.q0[:3]
    ready_normal = d.site_xmat[sid].reshape(3, 3)[:, 1].copy()

    self.ref = RefLib()
    # Seeded like everything else, so a match replays exactly from its seed.
    self.sides = [Side(n, self.ref, self.q_default, ready_blade, ready_normal, self.rng) for _ in range(2)]
    # Returns go to the back half of the other side, where the partner can play
    # them, keeping ~0.3 m (a typical landing error) inside the end line.
    self.plan_cfg = self.base_plan_cfg()
    self.launch = planner.LaunchCfg()
    # What the partner will go for, in _partner_ok's terms: the band its
    # planner chases, so a playable return is one it gets a plan for.
    self._reach_cfg = dataclasses.replace(self.plan_cfg, partner_y=(self.CHASE_Y[0] + 0.05, self.CHASE_Y[1] - 0.05),
                                          partner_z=self.CHASE_Z)
    nominal = BallParams.nominal(2048)
    p0, v0, w0, pl, ok = planner.serve_and_plan(self.rng, 2048, self.launch, planner.PlanCfg(), nominal, nominal)
    keep = np.flatnonzero(ok)
    self.bank = (p0[keep], v0[keep], w0[keep], planner.Plan(*(getattr(pl, f)[keep] for f in pl.__dataclass_fields__)))
    self.prm = BallParams.randomised(self.rng, n, dr_scale)
    self.prm_model = BallParams.nominal(n)

    self.bp = np.tile([0.0, 0.0, -5.0], (n, 1))
    self.bv = np.zeros((n, 3))
    self.bw = np.zeros((n, 3))
    self.live = np.zeros(n, bool)
    self.to = np.zeros(n, dtype=np.int64)  # side that must hit next
    self.bounced = np.zeros(n, bool)  # has the ball bounced on that side yet
    self.hitter = np.full(n, -1)  # last robot to hit; -1 after a serve
    self.next_receiver = self.rng.integers(0, 2, n)
    self.timer = self.rng.uniform(0.6, 1.2, n)
    self.t = np.zeros(n)
    self.ball_t = np.zeros(n)
    self.steps = np.zeros(n, dtype=np.int64)
    self.rally_len = np.zeros(n, dtype=np.int64)
    self.rally_wide = np.zeros(n, dtype=np.int64)  # hits in this rally on balls outside standing reach
    self.rally_moved = np.zeros(n, dtype=np.int64)  # hits made with the pelvis >= MOVED from the spot
    self.rally_fh = np.zeros(n, dtype=np.int64)  # hits in this rally on the forehand (red) face
    self.rally_bh = np.zeros(n, dtype=np.int64)  # ... and on the backhand (black) face
    # Returns landed before each backhand / each hit after moving, to tell the
    # ones the rally carried on from (see SOLID_AFTER) from the serve receive
    # and from a last shot the partner could not return.
    self.rally_bh_at = [[] for _ in range(n)]
    self.rally_mv_at = [[] for _ in range(n)]
    if sgs is not None:
      self._sgs_kw = {k: sgs[k] for k in ("target", "kappa", "temperature", "history") if k in sgs}
      self.sgs_serve = SuccessGuidedSampler(len(self.bank[0]), **self._sgs_kw)
      self.sgs_ret = SuccessGuidedSampler(4 * self.SGS_BINS ** 2, **self._sgs_kw)
      # The ball the robot to play next is facing (kind 1 serve, 2 return; -1 none),
      # whether it has bounced on that robot's side, and the one it just hit,
      # whose outcome is known once that return lands or the rally ends.
      self.recv_kind = np.full(n, -1)
      self.recv_idx = np.zeros(n, dtype=np.int64)
      self.recv_arrived = np.zeros(n, bool)
      self.hit_kind = np.full(n, -1)
      self.hit_idx = np.zeros(n, dtype=np.int64)
    self.last_face = np.zeros(n)  # face of the latest hit: +1 forehand, -1 backhand
    self.replan_due = np.zeros(n, bool)
    self.blade = np.zeros((2, n, 3))
    self.blade_v = np.zeros((2, n, 3))
    self.blade_n = np.zeros((2, n, 3))
    self.stats = {"rallies": 0, "rally_sum": 0, "rally_max": 0, "hits_a": 0, "hits_b": 0,
                  "end_fault": 0, "end_miss": 0, "end_fall": 0, "end_time": 0, "serves": 0}
    self.rally_log = []
    for i in range(n):
      self._reset_robots(i)
    self._read_blades()

  # ---------------------------------------------------------------- kinematics
  def _reset_robots(self, i):
    m, d = self.model, self.datas[i]
    mujoco.mj_resetData(m, d)
    qa = self.q0.copy()
    qb = self.q0.copy()
    qb[0:2] = -qb[0:2]
    qb[3:7] = [0.0, 0.0, 0.0, 1.0]
    d.qpos[:36], d.qpos[36:] = qa, qb
    d.ctrl[:29] = d.ctrl[29:] = self.q_default
    mujoco.mj_forward(m, d)
    for s in self.sides:
      s.clear(np.array([i]))
      s.last_action[i] = 0

  def _read_blades(self):
    for i, d in enumerate(self.datas):
      for s in range(2):
        self.blade[s, i] = d.site_xpos[self.blade_sid[s]]
        self.blade_n[s, i] = d.site_xmat[self.blade_sid[s]][[1, 4, 7]]

  def _state(self, s):
    """Base, yaw, gyro, gravity, joints of robot s in its own frame."""
    qa, va = slice(36 * s, 36 * s + 36), slice(35 * s, 35 * s + 35)
    q = np.stack([d.qpos[qa] for d in self.datas])
    qv = np.stack([d.qvel[va] for d in self.datas])
    base = q[:, 0:3] * (FLIP if s else 1.0)
    yaw = _yaw(q[:, 3:7]) + (np.pi if s else 0.0)
    R = np.stack([d.xmat[self.pelvis[s]].reshape(3, 3) for d in self.datas])
    return base, yaw, qv[:, 3:6], -R[:, 2, :], q[:, 7:], qv[:, 6:]

  def observe(self):
    obs = []
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(s)
      blade = self.blade[s] * (FLIP if s else 1.0)
      obs.append(self.sides[s].obs(self.t, base, yaw, gyro, grav, qj, qdj, blade))
    return np.concatenate(obs)  # (2n, dim): robot A rows, then robot B rows

  # ---------------------------------------------------------------- rally
  def _to_side(self, x, s):
    return x * FLIP if s else x

  def _plan_for(self, ids, side, bounces_left, keep_stroke=False):
    """Plan robot `side`'s hit of the current ball, in its frame."""
    if len(ids) == 0:
      return
    f = FLIP if side else 1.0
    # After the bounce a valid plan is only refined; a ball the robot had no
    # plan for stays unplayed.
    if bounces_left == 0:
      ids = ids[self.sides[side].plan_valid[ids]]
      if len(ids) == 0:
        return
    cfg = dataclasses.replace(self.plan_cfg, hit_y=self.CHASE_Y, hit_z=self.CHASE_Z)
    sd = self.sides[side]
    if bounces_left == 1:
      # Decide now where this robot's return will go.
      self._choose_aims(sd, ids)
    pl = self._plan_groups(ids, sd, cfg, f, bounces_left)
    self._plan_done(ids, side, pl, bounces_left, keep_stroke)

  def _plan_groups(self, ids, sd, cfg, f, bounces_left):
    """Plan group by group (aim, or SGS configuration) with g1tt.planner."""
    pl = None
    groups = sd.cell[ids] if self.sgs is not None else sd.aim[ids]
    for g in np.unique(groups):
      sub_ids = np.flatnonzero(groups == g)
      aim = int(g) // self.SGS_BINS ** 2 if self.sgs is not None else int(g)
      if aim in (1, 2):
        c = dataclasses.replace(cfg, partner_y=self.WIDE_Y[aim - 1], land_x=(0.30, 0.90),
                                land_y=self.WIDE_LAND_Y[aim - 1], land_retries=6)
      else:
        # The partner's frame is turned half a circle: landing on our +y side
        # reaches its forehand, our -y side its backhand.
        c = dataclasses.replace(cfg, partner_y=self.BH_Y if aim == 3 else self.FH_Y,
                                land_y=self.BH_LAND_Y if aim == 3 else self.FH_LAND_Y, land_retries=self.AIM_RETRIES)
      if self.sgs is not None:
        c = self._cell_ranges(c, int(g))
      j = ids[sub_ids]
      part = planner.plan_hits(self.bp[j] * f, self.bv[j] * f, self.bw[j] * f, self.prm_model.take(j),
                               c, self.rng, bounces_left=bounces_left, strict=bounces_left == 1,
                               horizon=0.6 if bounces_left == 0 else 1.6)
      # Backhand returns: planned higher over the net and deeper (BH_CLEAR).
      bh = np.flatnonzero(part.valid & (part.blade_pos[:, 1] > RefLib.BACKHAND_FROM_Y))
      if len(bh) and self.BH_CLEAR > c.net_clearance:
        hi = planner.plan_hits(self.bp[j[bh]] * f, self.bv[j[bh]] * f, self.bw[j[bh]] * f, self.prm_model.take(j[bh]),
                               dataclasses.replace(c, net_clearance=self.BH_CLEAR, land_x=self.BH_LAND_X), self.rng,
                               bounces_left=bounces_left, strict=bounces_left == 1,
                               horizon=0.6 if bounces_left == 0 else 1.6)
        ok = hi.valid & (hi.blade_pos[:, 1] > RefLib.BACKHAND_FROM_Y)
        for k in part.__dataclass_fields__:
          getattr(part, k)[bh[ok]] = getattr(hi, k)[ok]
      if pl is None:
        pl = planner.Plan(np.zeros(len(ids), bool), np.zeros(len(ids)), *(np.zeros((len(ids), 3)) for _ in range(7)))
      for k in pl.__dataclass_fields__:
        getattr(pl, k)[sub_ids] = getattr(part, k)
    return pl

  def _plan_done(self, ids, side, pl, bounces_left, keep_stroke):
    if bounces_left == 1:
      self.stats["unreachable"] = self.stats.get("unreachable", 0) + int((~pl.valid).sum())
      b = self.plan_cfg
      in_band = ((pl.ball_pos[:, 2] > b.hit_z[0]) & (pl.ball_pos[:, 2] < b.hit_z[1])
                 & (pl.ball_pos[:, 1] > b.hit_y[0]) & (pl.ball_pos[:, 1] < b.hit_y[1]))
      wide = pl.valid & ~in_band
      self.sides[side].wide[ids] = wide
      self.stats["wide_balls"] = self.stats.get("wide_balls", 0) + int(wide.sum())
    keep = pl.valid if keep_stroke else np.ones(len(ids), bool)
    if keep.any():
      sub = planner.Plan(*(getattr(pl, k)[keep] for k in pl.__dataclass_fields__))
      self.sides[side].set_plan(ids[keep], sub, self.t[ids[keep]], keep_stroke=keep_stroke)

  def _choose_aims(self, sd, ids):
    """Where these robots' returns will go: the partner's forehand, its
    backhand (bh_frac), or wide of its forehand (wide_frac)."""
    r = self.rng.random(len(ids))
    sd.aim[ids] = np.where(r < self.wide_frac, 1, np.where(r < self.wide_frac + self.bh_frac, 3, 0))
    if self.sgs is not None:
      # Every return is one of the SGS configurations: its aim and a landing
      # cell. A share `mix` is picked by SGS, the rest keep the usual aim with
      # a uniform cell, so their outcomes keep the estimates current.
      b2 = self.SGS_BINS ** 2
      cell = sd.aim[ids] * b2 + self.rng.integers(0, b2, len(ids))
      by_sgs = self.rng.random(len(ids)) < self.sgs.get("mix", 0.5)
      if by_sgs.any():
        cell[by_sgs] = self.sgs_ret.sample(int(by_sgs.sum()), self.rng)
      sd.cell[ids] = cell
      sd.aim[ids] = cell // b2

  def _cell_ranges(self, c, cell):
    """Narrow an aim's landing ranges to SGS configuration `cell` (depth x width thirds)."""
    k = self.SGS_BINS
    dx, dy = (cell % k ** 2) // k, cell % k
    third = lambda r, i: (r[0] + (r[1] - r[0]) * i / k, r[0] + (r[1] - r[0]) * (i + 1) / k)
    return dataclasses.replace(c, land_x=third(c.land_x, dx), land_y=third(c.land_y, dy))

  def _serve(self, ids):
    if len(ids) == 0:
      return
    p0, v0, w0, plan = self.bank
    j = self.rng.integers(0, len(p0), len(ids))
    if self.sgs is not None:
      by_sgs = self.rng.random(len(ids)) < self.sgs.get("mix", 0.5)
      if by_sgs.any():
        j[by_sgs] = self.sgs_serve.sample(int(by_sgs.sum()), self.rng)
      self.recv_kind[ids], self.recv_idx[ids], self.recv_arrived[ids] = 1, j, False
      self.hit_kind[ids] = -1
    recv = self.next_receiver[ids]
    f = np.where(recv[:, None] == 1, FLIP, 1.0)
    self.bp[ids], self.bv[ids], self.bw[ids] = p0[j] * f, v0[j] * f, w0[j] * f
    self.live[ids] = True
    self.to[ids], self.bounced[ids], self.hitter[ids] = recv, False, -1
    self.ball_t[ids] = 0.0
    self.next_receiver[ids] = 1 - recv
    self.stats["serves"] += len(ids)
    for s in range(2):
      self.sides[s].clear(ids)
      mine = ids[recv == s]
      if len(mine):
        pl = planner.Plan(*(getattr(plan, k)[j[recv == s]] for k in plan.__dataclass_fields__))
        self.sides[s].set_plan(mine, pl, self.t[mine])
        if self.sgs is not None:
          self._choose_aims(self.sides[s], mine)

  def _sgs_record(self, ids, success):
    """The configurations just played (hit_*) in matches `ids` succeeded or failed."""
    for kind, smp in ((1, self.sgs_serve), (2, self.sgs_ret)):
      k = ids[self.hit_kind[ids] == kind]
      if len(k):
        smp.record(self.hit_idx[k], np.full(len(k), success))
    self.hit_kind[ids] = -1

  def _sgs_end(self, ids, reason):
    """A rally ended: a return that did not land failed its configuration, and so
    did a ball the receiver had in front of it (missed, or fell). Balls that never
    reached the receiver, and time-outs, say nothing."""
    if reason != "time":
      self._sgs_record(ids[self.hit_kind[ids] >= 0], False)
      faced = ids[(self.recv_kind[ids] >= 0) & self.recv_arrived[ids]]
      self.hit_kind[faced], self.hit_idx[faced] = self.recv_kind[faced], self.recv_idx[faced]
      self._sgs_record(faced, False)
    self.hit_kind[ids] = self.recv_kind[ids] = -1
    self.recv_arrived[ids] = False

  def _end_rally(self, ids, reason):
    if len(ids) == 0:
      return
    if self.sgs is not None:
      self._sgs_end(ids, reason)
    self.stats["rallies"] += len(ids)
    self.stats["rally_sum"] += int(self.rally_len[ids].sum())
    self.stats["rally_max"] = max(self.stats["rally_max"], int(self.rally_len[ids].max()))
    self.stats[f"end_{reason}"] += len(ids)
    self.rally_log += [(int(i), float(self.t[i]), int(self.rally_len[i]), reason, int(self.rally_wide[i]),
                        int(self.rally_moved[i]), int(self.rally_fh[i]), int(self.rally_bh[i]),
                        self._solid(self.rally_bh_at[i], self.rally_len[i]),
                        self._solid(self.rally_mv_at[i], self.rally_len[i])) for i in ids]
    for i in ids:
      self.rally_bh_at[i], self.rally_mv_at[i] = [], []
    self.rally_len[ids] = 0
    self.rally_wide[ids] = 0
    self.rally_moved[ids] = 0
    self.rally_fh[ids] = 0
    self.rally_bh[ids] = 0
    self.live[ids] = False
    self.bp[ids] = [0.0, 0.0, -5.0]
    self.timer[ids] = self.rng.uniform(0.6, 1.2, len(ids))
    for s in self.sides:
      s.clear(ids)

  def _solid(self, at, length):
    """Shots in a rally of `length` returns that were not the serve receive
    and after which SOLID_AFTER more returns landed (theirs included)."""
    return sum(1 for k in at if k >= 1 and length >= k + 1 + self.SOLID_AFTER)

  def step(self, actions):
    fell, timeout = self._advance(actions)
    self._finish(fell, timeout)
    return self.observe()

  def _advance(self, actions):
    """Physics and rally rules for one control step. Fills self.ev with what
    happened, per robot, for rewards. Returns (fell (2, n), timeout (n))."""
    n = self.n
    z = lambda *sh: np.zeros((2, n) + sh)
    self.ev = ev = {
      "hit": z().astype(bool), "vout": z(3), "shortfall": z(), "land": z().astype(bool), "land_err": z(),
      "miss": z().astype(bool), "fault": z().astype(bool),
      "hit_face_ok": z().astype(bool), "land_face_ok": z().astype(bool),  # contact on the commanded face
      "imp": z().astype(bool), "imp_pos": z(), "imp_vel": z(), "imp_cos": z(),
      "playable": z().astype(bool),  # the return will reach the partner's hitting plane, in its reach
    }
    a = np.clip(actions, -4.0, 4.0).reshape(2, n, 29)
    for s in range(2):
      self.sides[s].prev_action[:] = self.sides[s].last_action
      self.sides[s].last_action[:] = a[s]
    targets = np.clip(self.q_default + self.act_scale * a, self.jnt_lo, self.jnt_hi)
    h = self.phys_dt / BALL_TICKS
    self._serve(np.flatnonzero(~self.live & (self.timer <= 0)))
    for _ in range(SUBSTEPS):
      c0, n0 = self.blade.copy(), self.blade_n.copy()
      for i, d in enumerate(self.datas):
        d.ctrl[:29], d.ctrl[29:] = targets[0, i], targets[1, i]
        mujoco.mj_step(self.model, d)
      self._read_blades()
      vpad = (self.blade - c0) / self.phys_dt
      wpad = ball.cross(n0, self.blade_n) / self.phys_dt
      t_prev = self.t.copy()
      self.t += self.phys_dt
      self.timer -= self.phys_dt
      idx = np.arange(n)
      for s in range(2):
        sd = self.sides[s]
        cross = sd.plan_valid & (t_prev < sd.t_hit) & (self.t >= sd.t_hit)
        if cross.any():
          f = FLIP if s else 1.0
          ev["imp"][s] |= cross
          ev["imp_pos"][s] = np.where(cross, np.linalg.norm(self.blade[s] * f - sd.p_star, axis=-1), ev["imp_pos"][s])
          ev["imp_vel"][s] = np.where(cross, np.linalg.norm(vpad[s] * f - sd.v_star, axis=-1), ev["imp_vel"][s])
          ev["imp_cos"][s] = np.where(cross, np.sum(self.blade_n[s] * f * sd.n_star, -1), ev["imp_cos"][s])
      self.blade_v = vpad
      for k in range(BALL_TICKS):
        a0, a1 = k / BALL_TICKS, (k + 1) / BALL_TICKS
        to = self.to
        C0 = c0[to, idx] + a0 * (self.blade[to, idx] - c0[to, idx])
        C1 = c0[to, idx] + a1 * (self.blade[to, idx] - c0[to, idx])
        N0 = n0[to, idx] + a0 * (self.blade_n[to, idx] - n0[to, idx])
        N1 = n0[to, idx] + a1 * (self.blade_n[to, idx] - n0[to, idx])
        p_prev = self.bp
        p, v, w, table, net, floor = ball.step_world(self.bp, self.bv, self.bw, h, self.prm, self.live)
        hit_done_to = np.where(to == 0, self.sides[0].hit_done, self.sides[1].hit_done)
        can_hit = self.live & self.bounced & ~hit_done_to
        p, v, w, hit, face_side = ball.paddle_contact(p_prev, p, v, w, C0, C1, N0, N1, vpad[to, idx],
                                                      wpad[to, idx], BLADE_RADIUS, can_hit, self.prm)
        if hit.any():
          sd_face = np.where(to == 0, self.sides[0].face, self.sides[1].face)
          for name, m in (("hits_fh", hit & (face_side > 0)), ("hits_bh", hit & (face_side < 0)),
                          ("cmd_bh", hit & (sd_face < 0)), ("cmd_bh_on_bh", hit & (sd_face < 0) & (face_side < 0)),
                          ("cmd_fh_on_bh", hit & (sd_face > 0) & (face_side < 0))):
            self.stats[name] = self.stats.get(name, 0) + int(m.sum())
          self.rally_fh += hit & (face_side > 0)
          self.rally_bh += hit & (face_side < 0)
          for i in np.flatnonzero(hit & (face_side < 0)):
            self.rally_bh_at[i].append(int(self.rally_len[i]))
          self.last_face = np.where(hit, face_side, self.last_face)
        self.bp, self.bv, self.bw = p, v, w
        self.ball_t += h * self.live

        half = (p[:, 0] > 0).astype(np.int64)  # 0: A's half, 1: B's half
        on_to = table & (half == to)
        landed = on_to & ~self.bounced & (self.hitter >= 0)
        self.rally_len += landed
        if self.sgs is not None and landed.any():
          self._sgs_record(np.flatnonzero(landed & (self.hit_kind >= 0)), True)
        for name, m in (("land_fh", landed & (self.last_face > 0)), ("land_bh", landed & (self.last_face < 0))):
          self.stats[name] = self.stats.get(name, 0) + int(m.sum())
        for s in range(2):
          ls = landed & (self.hitter == s)
          if ls.any():
            ev["land"][s] |= ls
            ev["land_face_ok"][s] |= ls & (self.last_face * self.sides[s].face > 0)
            pf = p * (FLIP if s else 1.0)
            ev["land_err"][s] = np.where(ls, np.linalg.norm(pf[:, :2] - self.sides[s].land_tgt[:, :2], axis=-1),
                                         ev["land_err"][s])
        first_bounce = on_to & ~self.bounced
        self.bounced |= first_bounce
        if self.sgs is not None:
          self.recv_arrived |= first_bounce
        self.replan_due |= first_bounce

        fault = self.live & ~self.bounced & (self.hitter >= 0) & ((table & (half != to)) | net | floor)
        void = self.live & ~self.bounced & (self.hitter < 0) & ((table & (half != to)) | net | floor)
        behind = np.where(to == 0, p[:, 0] < ROBOT_X - 0.6, p[:, 0] > -ROBOT_X + 0.6)
        miss = self.live & self.bounced & ~hit & ((on_to & ~first_bounce) | (table & (half != to)) | net | floor
                                                  | behind | (self.ball_t > 3.0))
        for s in range(2):
          ev["fault"][s] |= fault & (self.hitter == s)
          ev["miss"][s] |= miss & (to == s)
        for reason, mask in (("fault", fault), ("miss", miss), ("fault", void)):
          if mask.any():
            self._end_rally(np.flatnonzero(mask), reason)

        good = hit & self.live
        if good.any():
          ids = np.flatnonzero(good)
          for s in range(2):
            mine = ids[to[ids] == s]
            if len(mine):
              f = FLIP if s else 1.0
              ev["hit"][s, mine] = True
              ev["hit_face_ok"][s, mine] = face_side[mine] * self.sides[s].face[mine] > 0
              ev["vout"][s, mine] = v[mine] * f
              ev["shortfall"][s, mine] = ball.return_shortfall(p[mine] * f, v[mine] * f, w[mine] * f,
                                                               self.prm.take(mine))
              ev["playable"][s, mine] = planner._partner_ok(p[mine] * f, v[mine] * f, w[mine] * f,
                                                            self.prm.take(mine), self._reach_cfg, horizon=self.REACH_HORIZON)
          for s in range(2):
            mine = ids[to[ids] == s]
            self.sides[s].hit_done[mine] = True
            self.stats["wide_hits"] = self.stats.get("wide_hits", 0) + int(self.sides[s].wide[mine].sum())
            self.rally_wide[mine] += self.sides[s].wide[mine]
            if len(mine):
              base = np.stack([self.datas[i].qpos[36 * s:36 * s + 2] for i in mine]) * (FLIP[:2] if s else 1.0)
              moved = np.hypot(base[:, 0] - ROBOT_X, base[:, 1]) >= self.MOVED
              self.stats["moved_hits"] = self.stats.get("moved_hits", 0) + int(moved.sum())
              self.rally_moved[mine] += moved
              for i in mine[moved]:
                self.rally_mv_at[i].append(int(self.rally_len[i]))
            self.stats["hits_a" if s == 0 else "hits_b"] += len(mine)
          if self.sgs is not None:
            self.hit_kind[ids], self.hit_idx[ids] = self.recv_kind[ids], self.recv_idx[ids]
            cell = np.where(to[ids] == 0, self.sides[0].cell[ids], self.sides[1].cell[ids])
            self.recv_kind[ids] = np.where(cell >= 0, 2, -1)
            self.recv_idx[ids], self.recv_arrived[ids] = np.maximum(cell, 0), False
          self.hitter[ids] = to[ids]
          self.to[ids] = 1 - to[ids]
          self.bounced[ids] = False
          self.ball_t[ids] = 0.0
          for s in range(2):
            nxt = ids[self.to[ids] == s]
            self.sides[s].clear(nxt)
            self._plan_for(nxt, s, bounces_left=1)

    due = self.replan_due & self.live & self.bounced
    for s in range(2):
      ids = np.flatnonzero(due & (self.to == s) & ~self.sides[s].hit_done)
      self._plan_for(ids, s, bounces_left=0, keep_stroke=True)
    self.replan_due[:] = False

    self.steps += 1
    fell = np.zeros((2, n), bool)
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(s)
      fell[s] = (base[:, 2] < 0.5) | (grav[:, 2] > -0.7)
    timeout = self.steps >= self.max_steps
    return fell, timeout

  def _finish(self, fell2, timeout):
    """Reset matches where a robot fell or time ran out; update the display."""
    fell = fell2.any(0)
    for reason, mask in (("fall", fell), ("time", timeout & ~fell)):
      ids = np.flatnonzero(mask)
      if len(ids):
        self._end_rally(ids[self.live[ids] | (self.rally_len[ids] > 0)], reason)
        for i in ids:
          self._reset_robots(i)
        self.steps[ids] = 0
        self.on_reset(ids)
    self._read_blades()
    for i, d in enumerate(self.datas):
      d.mocap_pos[self.ball_mocap] = self.bp[i]
      for s in range(2):
        on = self.sides[s].plan_valid[i] and not self.sides[s].hit_done[i]
        d.mocap_pos[self.tgt_mocap[s]] = self._to_side(self.sides[s].p_star[i], s) if on else [0, 0, -5.0]

  def on_reset(self, ids):
    """Hook for subclasses: matches `ids` were just reset."""


def run(ckpt, matches, seconds, seed, dr_scale=1.0, video=None, width=1280, height=720, wide_frac=0.10,
        bh_frac=0.25):
  """Play `matches` matches for `seconds`; (summary, rally log). Deterministic
  policy and seeded, so the same arguments replay the same matches."""
  duel = Duel(matches, seed=seed, dr_scale=dr_scale, episode_s=seconds, wide_frac=wide_frac, bh_frac=bh_frac)
  ac = load_policy(ckpt, 2 * matches)
  obs = duel.observe()
  writer = renderer = None
  if video:
    m = duel.model
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, width), max(m.vis.global_.offheight, height)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 4.4, 90, -16
    Path(video).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video, fps=int(1 / CTRL_DT), quality=8)
  for _ in range(int(seconds / CTRL_DT)):
    a = ac.act_deterministic(torch.from_numpy(obs)).numpy()
    obs = duel.step(a)
    if writer:
      renderer.update_scene(duel.datas[0], cam)
      writer.append_data(renderer.render())
  if writer:
    writer.close()
  # Rallies still going at the end count with their length so far.
  ongoing = np.flatnonzero(duel.live | (duel.rally_len > 0))
  duel._end_rally(ongoing, "time")
  s = duel.stats
  lens = np.array([r[2] for r in duel.rally_log] or [0])
  longest = max(duel.rally_log, key=lambda r: r[2]) if duel.rally_log else None
  return {
    "matches": matches, "seconds": seconds, "serves": s["serves"], "rallies": s["rallies"],
    "rally_mean_returns": s["rally_sum"] / max(s["rallies"], 1), "rally_longest": s["rally_max"],
    "hits_a": s["hits_a"], "hits_b": s["hits_b"],
    "wide_balls": s.get("wide_balls", 0), "wide_hits": s.get("wide_hits", 0),
    "moved_hits": s.get("moved_hits", 0),
    **{k: s.get(k, 0) for k in ("hits_fh", "hits_bh", "land_fh", "land_bh", "cmd_bh", "cmd_bh_on_bh", "cmd_fh_on_bh")},
    "unreachable": s.get("unreachable", 0),
    "ends": {k[4:]: s[k] for k in s if k.startswith("end_")},
    "rally_percentiles_50_90_99": [float(np.percentile(lens, q)) for q in (50, 90, 99)],
    "rallies_ge_50": int((lens >= 50).sum()), "rallies_ge_100": int((lens >= 100).sum()),
    "longest_rally_footwork_hits": longest[4] if longest else 0,
    "longest_rally_moved_hits": longest[5] if longest else 0,
    "longest_rally_fh_bh_hits": list(longest[6:8]) if longest else [0, 0],
    "longest_rally_solid_bh_moved": list(longest[8:10]) if longest else [0, 0],
  }, duel.rally_log


def find_long_rally(ckpt, seeds, seconds, dr_scale=1.0, wide_frac=0.10, bh_frac=0.25):
  """One match per seed, no rendering; (seed, returns, end time, footwork hits)
  of the longest rally. Runs are deterministic, so filming the seed replays it."""
  best = (None, -1, 0.0, 0)
  for seed in seeds:
    res, log = run(ckpt, 1, seconds, seed, dr_scale, wide_frac=wide_frac, bh_frac=bh_frac)
    for _, t, k, why, wd, mv, *_ in log:
      if k > best[1]:
        best = (seed, k, t, wd)
  return best


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("ckpt")
  ap.add_argument("--matches", type=int, default=8)
  ap.add_argument("--seconds", type=float, default=120.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--dr-scale", type=float, default=1.0)
  ap.add_argument("--video", default=None, help="film match 0")
  ap.add_argument("--find-rally", type=int, default=0, help="search this many seeds (1 match each) and film the longest")
  ap.add_argument("--wide-frac", type=float, default=0.10, help="share of returns aimed wide, beyond standing reach")
  ap.add_argument("--bh-frac", type=float, default=0.25, help="share of returns aimed at the partner's backhand")
  args = ap.parse_args()
  if args.find_rally:
    seed, k, t, wd = find_long_rally(args.ckpt, range(args.seed, args.seed + args.find_rally), args.seconds,
                                     args.dr_scale, wide_frac=args.wide_frac, bh_frac=args.bh_frac)
    print(f"longest rally: {k} returns ({wd} with footwork), seed {seed}, ends at {t:.1f} s")
    args.seed, args.seconds, args.matches = seed, t + 1.0, 1
    args.video = args.video or f"renders/rally_{k}.mp4"
  res, log = run(args.ckpt, args.matches, args.seconds, args.seed, args.dr_scale, args.video,
                 wide_frac=args.wide_frac, bh_frac=args.bh_frac)
  print(json.dumps(res, indent=1))
  if args.video:
    mine = [r for r in log if r[0] == 0]
    print("match 0 rallies (end time s, returns, ended by, wide-ball hits, hits after moving, forehand, backhand, "
          "solid backhands, solid hits after moving):", [(round(r[1], 1), *r[2:]) for r in mine])


if __name__ == "__main__":
  main()
