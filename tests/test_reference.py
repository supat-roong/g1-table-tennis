import numpy as np
import pytest

from g1tt.mocap import OUT

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="run `python -m g1tt.mocap` after downloading TTMD6")


def test_reference_timing_puts_contact_on_the_planned_instant():
  from g1tt.reference import RefLib
  lib = RefLib()
  rng = np.random.default_rng(0)
  p = lib.imp_blade[:5] + 0.01
  v = lib.imp_vel[:5] * 0.5  # half the human speed: the stroke is slowed down
  clip, stretch = lib.choose(p, v, rng)
  assert np.all(stretch >= 1.0)
  q_imp = lib.joints(clip, np.zeros(5), stretch)
  assert np.allclose(q_imp, lib.q[lib.offsets[clip] + lib.impact[clip]])
  pre, post = lib.window(clip, stretch)
  q_start = lib.joints(clip, -pre, stretch)
  assert np.allclose(q_start, lib.q[lib.offsets[clip]])
