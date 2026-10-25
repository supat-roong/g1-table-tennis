"""Self-play training: both G1s in a match learn, with the same policy.

Each match yields two agents, robot A's row then robot B's, computed in that
robot's own frame (see g1tt.duel). The other robot's returns, not a scripted
opponent, are what each robot has to play. They are spread across the table
and some go wide, so rallies need footwork: the penalty for leaving the spot
only starts beyond `move_free` metres.
"""

import dataclasses

import numpy as np

from g1tt import ball, planner
from g1tt.ball import BallParams
from g1tt.duel import FLIP, Duel, _rot, _yaw_rot
from g1tt.env import CRITIC_DIM, EnvCfg, scaled_launch
from g1tt.model import ROBOT_X, TABLE_H, TABLE_LEN, TABLE_WID
from g1tt.reference import IMITATE, LEGS
from g1tt.sgs import SuccessGuidedSampler

WAIT, INCOMING, BOUNCED, RETURN = 0, 1, 2, 3


class DuelEnv(Duel):
  def __init__(self, num_envs: int, seed: int = 0, cfg: EnvCfg | None = None):
    assert num_envs % 2 == 0, "two robots per match"
    self.cfg = cfg or EnvCfg()
    super().__init__(num_envs // 2, seed=seed, dr_scale=self.cfg.dr_scale, episode_s=self.cfg.episode_s,
                     wide_frac=self.cfg.wide_frac, bh_frac=self.cfg.bh_frac, sgs=self.cfg.sgs)
    self.num_agents = num_envs
    # Returns go deep, spread across the other robot's half.
    self.plan_cfg = dataclasses.replace(self.plan_cfg, land_y=self.cfg.land_y)
    self.set_level(self.cfg.level)
    self.steps[:] = self.rng.integers(0, self.max_steps, self.n)  # desynchronise episodes
    self.rally_reward_len = np.zeros(self.n)
    self.tot = dict.fromkeys(
      ["balls", "hits", "lands", "faults", "misses", "falls", "episodes", "imp_pos_err", "imp_n", "land_err",
       "ref_rmse", "ref_n", "rallies", "rally_sum", "rally_max", "shot_err", "shortfall", "bh_hits",
       "bh_playable"], 0.0)

  def set_level(self, level: float):
    self.cfg.level = level
    nominal = BallParams.nominal(2048)
    launch = scaled_launch(self.cfg.launch, level)
    p0, v0, w0, pl, ok = planner.serve_and_plan(self.rng, 2048, launch, planner.PlanCfg(), nominal, nominal)
    keep = np.flatnonzero(ok)
    self.bank = (p0[keep], v0[keep], w0[keep], planner.Plan(*(getattr(pl, f)[keep] for f in pl.__dataclass_fields__)))
    if self.sgs is not None:  # a new serve bank: start its success estimates afresh
      self.sgs_serve = SuccessGuidedSampler(len(self.bank[0]), **self._sgs_kw)

  # ---------------------------------------------------------------- observations
  def observe(self):
    obs = super().observe()
    if self.cfg.obs_noise:
      obs = obs + self.cfg.obs_noise * self.rng.standard_normal(obs.shape).astype(np.float32)
    return obs

  def _phase(self, s):
    ph = np.full(self.n, WAIT)
    mine = self.live & (self.to == s)
    ph[mine & ~self.bounced] = INCOMING
    ph[mine & self.bounced] = BOUNCED
    ph[self.live & (self.to != s)] = RETURN
    return ph

  def critic_obs(self, obs=None):
    obs = super().observe() if obs is None else obs
    out = []
    for s in range(2):
      f = FLIP if s else 1.0
      base, yaw, gyro, grav, qj, qdj = self._state(s)
      R = _yaw_rot(yaw)
      lin = np.stack([d.qvel[35 * s:35 * s + 3] for d in self.datas]) * f
      prm = self.prm
      phys = np.stack([prm.kd / 0.126, prm.km / 2.8e-3, prm.table_e, prm.table_mu, prm.rubber_e, prm.rubber_mu], -1)
      alive = self.live[:, None]
      ball_rel = np.where(alive, _rot(R, self.bp * f - base), 0.0)
      priv = np.concatenate([
        _rot(R, lin), base[:, 2:3], ball_rel, 0.2 * _rot(R, self.bv * f) * alive, 0.005 * self.bw * f * alive,
        _rot(R, self.blade[s] * f - base), 0.3 * _rot(R, self.blade_v[s] * f), phys, np.eye(4)[self._phase(s)],
      ], -1)
      # Actor obs except its trailing reference/home/path blocks, then
      # privileged state, then those blocks.
      a = obs[s * self.n:(s + 1) * self.n]
      head, tail = a[:, :3 + 3 + 29 * 3 + 11], a[:, 3 + 3 + 29 * 3 + 11:]
      out.append(np.concatenate([head, priv, tail], -1).astype(np.float32))
    c = np.concatenate(out)
    assert c.shape[1] == CRITIC_DIM
    return c

  # ---------------------------------------------------------------- rewards
  def _rewards(self, fell2):
    rc, n, ev = self.cfg.reward, self.n, self.ev
    rew, terms_all = [], {}
    for s in range(2):
      sd = self.sides[s]
      f = FLIP if s else 1.0
      base, yaw, gyro, grav, qj, qdj = self._state(s)
      blade = self.blade[s] * f
      bn = self.blade_n[s] * f
      bv = self.blade_v[s] * f
      torque = np.stack([d.actuator_force[29 * s:29 * s + 29] for d in self.datas])
      tau = sd.t_hit - self.t
      my_turn = self.live & (self.to == s)
      active = sd.plan_valid & ~sd.hit_done & (tau > -0.06) & my_turn
      T = {}

      r_act, rel = sd._window(self.t, 0.0)
      r = np.where(r_act, rel, 0.0)
      pt = sd.ref.blade_path(sd.ref_clip, r, sd.ref_stretch, sd.ref_shift)
      v_h = sd.ref.imp_vel[sd.ref_clip] / sd.ref_stretch[:, None]
      pt = pt + (sd.v_star - v_h) * (r * np.exp(-(r / 0.15) ** 2))[:, None] + [ROBOT_X, 0.0, 0.0]
      T["path"] = rc.path * np.exp(-(np.linalg.norm(blade - pt, axis=-1) / 0.10) ** 2) * r_act
      q_ref = sd.ref.joints(sd.ref_clip, r, sd.ref_stretch)
      imitate = r_act & (rel <= rc.ref_until)
      e_up = np.mean((qj[:, IMITATE] - q_ref[:, IMITATE]) ** 2, -1)
      e_leg = np.mean((qj[:, LEGS] - q_ref[:, LEGS]) ** 2, -1)
      T["ref_upper"] = rc.ref_upper * np.exp(-e_up / 0.5) * imitate
      T["ref_legs"] = rc.ref_legs * np.exp(-e_leg / 0.3) * imitate
      e_wr = np.mean((qj[:, self.WRIST] - q_ref[:, self.WRIST]) ** 2, -1)
      T["ref_wrist"] = rc.ref_wrist * np.exp(-e_wr / 0.5) * imitate * (sd.face < 0)
      self.tot["ref_rmse"] += np.sqrt(e_up[imitate]).sum()
      self.tot["ref_n"] += imitate.sum()

      cos_face = np.sum(bn * sd.n_star, -1)  # signed: the commanded face, forehand or backhand
      T["face"] = rc.face * np.exp(-(1 - cos_face) / 0.05) * (active & (tau > -0.02) & (tau < 0.12))
      T["face_coarse"] = rc.face_coarse * (1 - np.arccos(np.clip(cos_face, -1, 1)) / np.pi) * (
        active & (tau > -0.02) & (tau < 0.35))
      T["blade_vel"] = rc.blade_vel * np.exp(-(np.linalg.norm(bv - sd.v_star, axis=-1) / 0.4) ** 2) * (
        active & (tau > -0.03) & (tau < 0.08))
      imp = ev["imp"][s]
      T["impact"] = imp * (rc.impact_pos * np.exp(-(ev["imp_pos"][s] / 0.05) ** 2)
                           + rc.impact_pos_coarse * np.exp(-(ev["imp_pos"][s] / 0.15) ** 2)
                           + rc.impact_vel * np.exp(-(ev["imp_vel"][s] / 1.0) ** 2)
                           + rc.impact_ori * np.exp(-(1 - ev["imp_cos"][s]) / 0.1))
      self.tot["imp_pos_err"] += ev["imp_pos"][s][imp].sum()
      self.tot["imp_n"] += imp.sum()

      # Only a hit on the commanded face scores: a backhand ball met with the
      # forehand face (paddle not turned over) earns nothing, though it plays on.
      hit = ev["hit"][s] & ev["hit_face_ok"][s]
      shot_err = np.linalg.norm(ev["vout"][s] - sd.vout_star, axis=-1)
      T["hit"] = rc.hit * hit
      T["shot"] = rc.shot * np.exp(-(shot_err / 1.0) ** 2) * hit
      T["shot_outcome"] = rc.shot_outcome * np.exp(-(ev["shortfall"][s] / 0.25) ** 2) * hit
      T["playable"] = rc.playable * (ev["playable"][s] & hit)
      bh_hit = hit & (sd.face < 0)
      self.tot["bh_hits"] += bh_hit.sum()
      self.tot["bh_playable"] += (bh_hit & ev["playable"][s]).sum()
      self.tot["shot_err"] += shot_err[hit].sum()
      self.tot["shortfall"] += ev["shortfall"][s][hit].sum()
      land = ev["land"][s] & ev["land_face_ok"][s]
      T["land"] = rc.land * land + rc.land_acc * np.exp(-(ev["land_err"][s] / 0.3) ** 2) * land
      bh_scale = np.where(sd.face < 0, rc.bh_shot_scale, 1.0)
      for k in ("shot", "shot_outcome", "land"):
        T[k] = T[k] * bh_scale

      ready = np.exp(-(np.linalg.norm(blade - (base + sd.ready_blade), axis=-1) / 0.2) ** 2)
      T["ready"] = rc.ready * ready * ~active * ~r_act
      T["upright"] = rc.upright * np.exp(-np.sum(grav[:, :2] ** 2, -1) / 0.05)
      T["alive"] = np.full(n, rc.alive)
      # Footwork: while a hit is planned, move the pelvis to where the target
      # sits at the ready stance's reach (within 0.8 m of the spot); between
      # shots, drifting from the spot is penalised so the robot comes back.
      stance = sd.p_star[:, :2] - sd.ready_blade[:2]
      off = stance - [ROBOT_X, 0.0]
      off_n = np.linalg.norm(off, axis=-1, keepdims=True)
      stance = [ROBOT_X, 0.0] + off * np.minimum(1.0, 0.8 / np.maximum(off_n, 1e-6))
      moving = active & (tau > 0.0) & (tau < 0.9)
      T["position"] = rc.position * np.exp(-(np.linalg.norm(base[:, :2] - stance, axis=-1) / 0.25) ** 2) * moving
      drift = np.hypot(base[:, 0] - ROBOT_X, base[:, 1])
      idle = ~(my_turn & sd.plan_valid & ~sd.hit_done)
      T["drift"] = rc.drift * np.maximum(drift - self.cfg.move_free, 0.0) ** 2 * idle
      T["action_rate"] = rc.action_rate * np.sum((sd.last_action - sd.prev_action) ** 2, -1)
      T["torque"] = rc.torque * np.sum(torque ** 2, -1)
      T["leg_vel"] = rc.leg_vel * np.sum(qdj[:, :12] ** 2, -1)
      span = self.jnt_hi - self.jnt_lo
      lo, hi = self.jnt_lo + 0.05 * span, self.jnt_hi - 0.05 * span
      T["joint_limit"] = rc.joint_limit * np.sum(np.maximum(lo - qj, 0) + np.maximum(qj - hi, 0), -1)
      over = (np.abs(blade[:, 0]) < TABLE_LEN / 2) & (np.abs(blade[:, 1]) < TABLE_WID / 2)
      T["paddle_table"] = rc.paddle_table * (over & (blade[:, 2] < TABLE_H + 0.03))
      T["fall"] = rc.fall * fell2[s]
      rew.append(sum(T.values()))
      for k, v in T.items():
        terms_all[k] = terms_all.get(k, 0.0) + float(v.mean()) / 2

      self.tot["hits"] += hit.sum()
      self.tot["lands"] += land.sum()
      self.tot["land_err"] += ev["land_err"][s][land].sum()
      self.tot["faults"] += ev["fault"][s].sum()
      self.tot["misses"] += ev["miss"][s].sum()
    return np.concatenate(rew).astype(np.float32), terms_all

  # ---------------------------------------------------------------- step
  def step(self, actions):
    fell2, timeout = self._advance(actions)
    rew, terms = self._rewards(fell2)
    obs_pre = super().observe()
    term_cobs = self.critic_obs(obs_pre)
    # A fall ends the match for both robots: the one that fell is terminal,
    # the other is cut off (bootstrapped like a time-out).
    match_done = fell2.any(0) | timeout
    fell = np.concatenate([fell2[0], fell2[1]])
    cut = np.concatenate([match_done, match_done]) & ~fell
    self.tot["falls"] += fell.sum()
    self.tot["episodes"] += (fell | cut).sum()
    self._finish(fell2, timeout)
    obs = self.observe()
    cobs = self.critic_obs()
    return obs, cobs, rew, fell, {"terms": terms, "timeout": cut, "term_cobs": term_cobs}

  def on_reset(self, ids):
    self.steps[ids] = 0

  def pop_stats(self):
    s = dict(self.tot)
    s["balls"] = float(self.stats["serves"]) + s["hits"]
    s["rallies"] = float(self.stats["rallies"])
    s["rally_sum"] = float(self.stats["rally_sum"])
    s["rally_max"] = float(self.stats["rally_max"])
    for k in self.tot:
      self.tot[k] = 0.0
    s["bh_cmd"] = float(self.stats.get("cmd_bh", 0))
    s["bh_on_bh"] = float(self.stats.get("cmd_bh_on_bh", 0))
    if self.sgs is not None:  # summed over workers by VecEnv; train._rates averages them
      s["sgs_workers"] = 1.0
      for name, smp in (("serve", self.sgs_serve), ("ret", self.sgs_ret)):
        for k, v in smp.summary().items():
          s[f"sgs_{name}_{k}"] = v
    for k in ("serves", "rallies", "rally_sum", "rally_max", "cmd_bh", "cmd_bh_on_bh"):
      self.stats[k] = 0
    return s
