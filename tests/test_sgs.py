import numpy as np

from g1tt.sgs import SuccessGuidedSampler


def _with_rates(rates, history=100, **kw):
  smp = SuccessGuidedSampler(len(rates), history=history, **kw)
  for i, r in enumerate(rates):
    k = int(round(r * history))
    smp.record(np.full(history, i), np.r_[np.ones(k), np.zeros(history - k)])
  return smp


def test_peaks_at_target_and_keeps_all_configurations():
  smp = _with_rates([0.0, 0.25, 0.5, 0.75, 1.0], target=0.5, kappa=5.0, temperature=2.0)
  p = smp.probs()
  assert np.argmax(p) == 2
  assert p[1] > p[0] and p[3] > p[4]
  assert (p > 0).all()
  assert abs(p.sum() - 1.0) < 1e-9


def test_target_shifts_the_peak():
  smp = _with_rates([0.3, 0.5, 0.7], target=0.66, kappa=5.0)
  assert np.argmax(smp.probs()) == 2


def test_history_is_a_rolling_window():
  smp = SuccessGuidedSampler(1, history=4)
  smp.record([0, 0, 0, 0], [False] * 4)
  assert smp.success_rate()[0] == 0.0
  smp.record([0, 0, 0, 0], [True] * 4)
  assert smp.success_rate()[0] == 1.0


def test_unvisited_count_as_at_target():
  smp = SuccessGuidedSampler(3, target=0.5)
  smp.record([0] * 10, [True] * 10)
  p = smp.probs()
  assert p[1] > p[0] and p[2] > p[0]
  assert smp.summary()["visited"] == 1 / 3


def test_sample_follows_probs():
  smp = _with_rates([0.0, 0.5, 1.0], kappa=10.0, temperature=1.0)
  draws = smp.sample(4000, np.random.default_rng(0))
  assert np.bincount(draws, minlength=3).argmax() == 1
