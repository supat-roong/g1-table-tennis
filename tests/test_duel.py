import numpy as np

from g1tt.duel import Duel
from g1tt.duel_env import DuelEnv
from g1tt.env import ACTOR_DIM, CRITIC_DIM, EnvCfg


def test_robot_b_sees_what_robot_a_would():
  """In the ready pose with no ball, B's observation equals A's: B's frame is
  A's turned half a circle, so one policy drives both."""
  duel = Duel(2, seed=0)
  ob = duel.observe()
  assert ob.shape == (4, ACTOR_DIM)
  assert np.abs(ob[:2] - ob[2:]).max() < 1e-6


def test_self_play_env_steps():
  env = DuelEnv(4, seed=0, cfg=EnvCfg())
  o, c = env.observe(), env.critic_obs()
  assert o.shape == (4, ACTOR_DIM) and c.shape == (4, CRITIC_DIM)
  for _ in range(30):
    o, c, r, fell, info = env.step(np.zeros((4, 29)))
    assert np.isfinite(o).all() and np.isfinite(c).all() and np.isfinite(r).all()
    assert r.shape == (4,) and fell.shape == (4,) and info["timeout"].shape == (4,)
