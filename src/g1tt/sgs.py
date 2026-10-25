"""Success Guided Sampling (Zhang et al., "A Balanced Data Diet", CoRL 2026,
arXiv:2610.12465): spend training on the task configurations the policy can
almost do.

Each of N fixed configurations keeps its last `history` outcomes; its success
rate p scores it with a Beta-shaped kernel peaking at `target`,

    w = (p + eps)^(kappa * target) * (1 - p + eps)^(kappa * (1 - target)),

and the next configuration is drawn from softmax(log(max(w, eps) + eps) / T),
T >= 1. Every configuration keeps a nonzero probability, so estimates for
ones it cannot do yet (or has mastered) stay fresh. Unvisited configurations
count as being at the target, so they are tried early.
"""

import numpy as np


class SuccessGuidedSampler:
  def __init__(self, n: int, history: int = 100, target: float = 0.5, kappa: float = 5.0,
               temperature: float = 2.0, eps: float = 1e-4):
    self.n, self.history = n, history
    self.target, self.kappa, self.temperature, self.eps = target, kappa, temperature, eps
    self.outcomes = np.zeros((n, history), dtype=np.int8)
    self.count = np.zeros(n, dtype=np.int64)  # outcomes recorded so far (capped at history)
    self.next = np.zeros(n, dtype=np.int64)  # ring-buffer write position

  def success_rate(self) -> np.ndarray:
    seen = np.minimum(self.count, self.history)
    p = self.outcomes.sum(1) / np.maximum(seen, 1)
    return np.where(seen > 0, p, self.target)

  def probs(self) -> np.ndarray:
    p, k, t, e = self.success_rate(), self.kappa, self.target, self.eps
    w = (p + e) ** (k * t) * (1.0 - p + e) ** (k * (1.0 - t))
    logit = np.log(np.maximum(w, e) + e) / max(self.temperature, 1.0)
    z = np.exp(logit - logit.max())
    return z / z.sum()

  def sample(self, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(self.n, size=k, p=self.probs())

  def record(self, idx, success) -> None:
    for i, s in zip(np.atleast_1d(idx), np.atleast_1d(success)):
      self.outcomes[i, self.next[i]] = bool(s)
      self.next[i] = (self.next[i] + 1) % self.history
      self.count[i] += 1

  def summary(self) -> dict:
    seen = self.count > 0
    p = self.success_rate()
    return {"visited": float(seen.mean()), "success": float(p[seen].mean()) if seen.any() else 0.0,
            "frontier": float(((p > 0.2) & (p < 0.8) & seen).mean())}
