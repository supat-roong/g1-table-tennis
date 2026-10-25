"""Timing, observation sizes and the training configuration.

The policy sees proprioception plus a paddle command from the planner (target
blade position, face normal, velocity, time to impact), never the ball, as in
HITTER; the critic also sees the ball and the physics constants.

Timing: control at 50 Hz, MuJoCo at 250 Hz (4 ms), ball at 1 kHz with the
blade pose interpolated across each MuJoCo step.
"""

from dataclasses import dataclass, field

import numpy as np

from g1tt import planner
from g1tt.model import JOINTS
from g1tt.reference import UPPER

CTRL_DT = 0.02
SUBSTEPS = 5
BALL_TICKS = 4

REF_DIM = len(UPPER) + 1  # reference waist+arm pose 0.1 s ahead, and whether one is active
HOME_DIM = 2  # base offset from its spot, as motion capture would give it
PATH_DIM = 3  # where the paddle path wants the blade 0.1 s ahead, in the base frame
ACTOR_DIM = 3 + 3 + 29 * 3 + 11 + REF_DIM + HOME_DIM + PATH_DIM
CRITIC_DIM = ACTOR_DIM + 3 + 1 + 9 + 6 + 6 + 4
REF_LOOKAHEAD = 0.1
HIDDEN = np.array([0.0, 0.0, -5.0])


@dataclass
class RewardCfg:
  path: float = 3.0  # follow the human paddle path shifted onto the target
  face: float = 3.0  # face normal towards the planned one in the last 0.12 s
  # 1 - angle / pi between the face and the commanded one over the last
  # 0.35 s. The sharp face term is ~0 with the wrong face out, so on its own it
  # never teaches turning the paddle over for a backhand; linear in the angle
  # it still pulls when the face is fully reversed.
  face_coarse: float = 2.0
  blade_vel: float = 3.0  # blade velocity towards the planned one around contact
  impact_pos_coarse: float = 3.0  # sigma 15 cm
  impact_pos: float = 5.0  # sigma 5 cm: the blade disc is 7.6 cm in radius
  impact_vel: float = 2.0
  impact_ori: float = 2.0
  hit: float = 10.0
  shot: float = 6.0  # at the hit: ball's outgoing velocity vs the one planned to land on target
  shot_outcome: float = 10.0  # at the hit: graded "will it land": exp(-(shortfall / 0.25 m)^2)
  land: float = 15.0
  land_acc: float = 5.0
  ref_upper: float = 0.75  # follow the human stroke: waist and arms
  ref_legs: float = 0.15
  # Backhands: the right wrist follows the human stroke (roll, pitch, yaw).
  # The retargeted wrists turn the black face to the ball in every backhand
  # clip; on forehands they are an IK by-product that sets the face wrong, so
  # they stay free there.
  ref_wrist: float = 1.0
  ref_until: float = -0.05  # stop imitating this long before impact; the hit itself is the task's
  ready: float = 0.5  # back to the ready stance between balls
  upright: float = 0.2
  alive: float = 0.2
  drift: float = -10.0
  action_rate: float = -0.002
  torque: float = -5e-6
  leg_vel: float = -2e-4
  joint_limit: float = -1.0
  paddle_table: float = -1.0
  fall: float = -10.0
  # shot, shot_outcome and land count this many times on a backhand, whose
  # returns otherwise scatter: net, long, or too high for the partner.
  bh_shot_scale: float = 2.0
  # At the hit: the ball will reach the partner's hitting plane after one
  # bounce, inside the band it chases. Landing alone also pays for short
  # backhands that die mid-table, unplayable.
  playable: float = 10.0
  position: float = 3.0  # footwork: pelvis towards where the planned hit is in easy reach


def default_sgs() -> dict:
  """Success Guided Sampling (g1tt.sgs) of serves and return configurations."""
  return {"target": 0.5, "kappa": 5.0, "temperature": 2.0, "history": 100, "mix": 0.5}


@dataclass
class EnvCfg:
  episode_s: float = 10.0  # a training match; a fall ends it early for both robots
  dr_scale: float = 1.0  # ball/rubber/table randomisation strength
  obs_noise: float = 0.01
  level: float = 1.0  # serve difficulty, 0 slow and nearly spinless .. 1 full range
  desync: bool = True  # random start on the match clock (GPU training; the CPU one always spreads them)
  move_free: float = 0.10  # metres from the spot before drifting is penalised between shots
  land_y: tuple = (-0.45, 0.55)  # where returns aim across the other half, hitter's frame
  wide_frac: float = 0.10  # share of returns aimed wide of the partner's reach (g1tt.duel.Duel)
  bh_frac: float = 0.25  # share aimed at the partner's backhand, the rest at its forehand
  # Success Guided Sampling of serves and returns (g1tt.sgs); None for uniform.
  sgs: dict | None = field(default_factory=default_sgs)
  launch: planner.LaunchCfg = field(default_factory=planner.LaunchCfg)
  reward: RewardCfg = field(default_factory=RewardCfg)


def scaled_launch(base: planner.LaunchCfg, level: float) -> planner.LaunchCfg:
  """Serve curriculum: less spin and slower flights at low levels."""
  level = float(np.clip(level, 0.0, 1.0))
  spin = 0.2 + 0.8 * level
  return planner.LaunchCfg(
    start_x=base.start_x, start_y=base.start_y, start_z=base.start_z,
    bounce_x=base.bounce_x, bounce_y=base.bounce_y,
    flight_t=(base.flight_t[0] + 0.15 * (1 - level), base.flight_t[1] + 0.10 * (1 - level)),
    topspin=(base.topspin[0] * spin, base.topspin[1] * spin),
    sidespin=(base.sidespin[0] * level, base.sidespin[1] * level),
  )


def _action_scale():
  s = np.full(29, 0.25)
  for i, j in enumerate(JOINTS):
    if "shoulder" in j or "elbow" in j or "wrist" in j:
      s[i] = 0.5 if j.startswith("right") else 0.25
  return s
