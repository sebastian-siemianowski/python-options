"""
TECD Drift Model v3
-------------------
Theory
~~~~~~
TECD-v3 is a deterministic, signal-driven drift model designed for regime-adaptive
return prediction. It couples an evidence accumulator with entropy-derived tension,
then applies a constrained drift update with jump augmentation. Likelihood is
Gaussian with EWMA volatility and is intentionally simple (misspecified) so models
can be compared via LL / AIC / BIC / PIT against alternatives (e.g., Kalman).

Key scientific notes
- Drift is signal-driven (not latent): μ evolves from evidence/entropy, not as a
  hidden state.
- Likelihood is misspecified but consistent for relative comparison.
- PIT is a heuristic diagnostic for calibration, not a guarantee.
- The update is non-smooth in parameter space (entropy histogram + jump trigger).
- Designed for regime-adaptive drift detection with explicit falsifiability via
  LL/AIC/BIC/PIT.

Assumptions & limitations
- Uses pure numpy/scipy; no pandas, no randomness (deterministic given inputs).
- Entropy uses a fixed 15-bin histogram; windowed over normalized returns.
- Volatility uses EWMA on raw returns with fixed decay (lambda_vol) shared for
  normalization and likelihood variance. This is a modeling choice; tune as
  needed but math stays unchanged.
- Stability enforced by clamping μ, E, J and entropy bounds; violations receive
  heavy likelihood penalties.

Minimal usage
    from tecd_drift_v3 import TECDDriftModelV3
    import numpy as np
    r = np.random.default_rng(0).normal(0, 0.01, size=500)
    model = TECDDriftModelV3()
    model.fit(r)
    ll = model.log_likelihood(r)
    mu = model.predict_mu()
    pit = model.pit(r)

The class prints parameter settings on initialization for transparency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.stats import norm


class DriftModel:
    """Interface for drift models used in model selection."""

    def fit(self, returns: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def log_likelihood(self, returns: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def predict_mu(self) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def pit(self, returns: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class TECDParams:
    lambda_e: float = 0.94
    alpha: float = 1.0
    kappa: float = 0.05
    gamma: float = 0.05
    beta: float = 0.02
    eta: float = 0.25
    tau: float = 0.5
    entropy_window: int = 50
    max_mu: float = 0.2
    E_max: float = 10.0
    J_max: float = 1.0
    eps: float = 1e-8
    lambda_vol: float = 0.94  # EWMA decay for volatility


class TECDDriftModelV3(DriftModel):
    """TECD-v3 drift model (deterministic, histogram entropy, jump tension)."""

    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        p = TECDParams()
        if params:
            for k, v in params.items():
                if hasattr(p, k):
                    setattr(p, k, float(v))
        p.entropy_window = int(p.entropy_window)
        self.params = p
        self._mu: np.ndarray = np.array([])
        self._E: np.ndarray = np.array([])
        self._H: np.ndarray = np.array([])
        self._T: np.ndarray = np.array([])
        self._J: np.ndarray = np.array([])
        self._sigma: np.ndarray = np.array([])
        self._r_tilde: np.ndarray = np.array([])
        self._ll: Optional[float] = None
        print(
            f"TECD-v3 drift: λ={p.lambda_e:.3f} α={p.alpha:.3f} κ={p.kappa:.3f} "
            f"γ={p.gamma:.3f} β={p.beta:.3f} η={p.eta:.3f} τ={p.tau:.3f} W={p.entropy_window}"
        )

    # Public API --------------------------------------------------------------
    def fit(self, returns: np.ndarray) -> None:
        r = np.asarray(returns, dtype=float)
        n = r.size
        if n == 0:
            self._reset_states()
            self._ll = -1e12
            return

        p = self.params
        self._mu = np.zeros(n, dtype=float)
        self._E = np.zeros(n, dtype=float)
        self._H = np.zeros(n, dtype=float)
        self._T = np.zeros(n, dtype=float)
        self._J = np.zeros(n, dtype=float)
        self._sigma = np.zeros(n, dtype=float)
        self._r_tilde = np.zeros(n, dtype=float)

        # EWMA volatility warm-start from first ~10 samples (documented heuristic)
        var_init = max(np.var(r[: min(10, n)]), float(np.mean(r[: min(10, n)] * r[: min(10, n)])), p.eps)
        var = var_init
        lam_v = p.lambda_vol

        H_prev = 0.0
        mu_prev = 0.0
        E_prev = 0.0

        for t in range(n):
            rt = r[t]
            var = lam_v * var + (1.0 - lam_v) * (rt * rt)
            sigma_t = float(np.sqrt(max(var, p.eps)))
            self._sigma[t] = sigma_t

            r_tilde = rt / max(sigma_t, p.eps)
            self._r_tilde[t] = r_tilde

            if p.lambda_e == 0.0:
                E_t = np.sign(r_tilde) * (abs(r_tilde) ** p.alpha)
            else:
                E_t = p.lambda_e * E_prev + np.sign(r_tilde) * (abs(r_tilde) ** p.alpha)
            E_t = float(np.clip(E_t, -p.E_max, p.E_max))

            H_t = self._compute_entropy(self._r_tilde, t, p.entropy_window, p.eps)
            H_t = float(np.clip(H_t, 0.0, 1.0))

            T_t = abs(E_t) * H_t
            if T_t > p.tau:
                J_t = float(np.clip(p.eta * E_t, -p.J_max, p.J_max))
            else:
                J_t = 0.0

            mu_t = mu_prev + p.kappa * np.tanh(E_t) - p.gamma * (H_t - H_prev) - p.beta * mu_prev + J_t
            mu_t = float(np.clip(mu_t, -p.max_mu, p.max_mu))

            # Store
            self._E[t] = E_t
            self._H[t] = H_t
            self._T[t] = T_t
            self._J[t] = J_t
            self._mu[t] = mu_t

            # Prepare for next step
            E_prev = E_t
            H_prev = H_t
            mu_prev = mu_t

        self._ll = self._compute_log_likelihood(r)

    def log_likelihood(self, returns: np.ndarray) -> float:
        if self._ll is None or self._mu.size != len(returns):
            self.fit(returns)
        assert self._ll is not None
        return self._ll

    def predict_mu(self) -> np.ndarray:
        return self._mu.copy()

    def pit(self, returns: np.ndarray) -> np.ndarray:
        if self._mu.size != len(returns):
            self.fit(returns)
        z = (np.asarray(returns, dtype=float) - self._mu) / np.maximum(self._sigma, self.params.eps)
        pit_vals = norm.cdf(z)
        return np.clip(pit_vals, 1e-8, 1.0 - 1e-8)

    # Internal helpers -------------------------------------------------------
    def _reset_states(self) -> None:
        self._mu = np.array([])
        self._E = np.array([])
        self._H = np.array([])
        self._T = np.array([])
        self._J = np.array([])
        self._sigma = np.array([])
        self._r_tilde = np.array([])

    def _compute_entropy(self, r_tilde: np.ndarray, t: int, window: int, eps: float) -> float:
        start = max(0, t - int(window) + 1)  # inclusive rolling window of size ≤ W
        window_vals = r_tilde[start : t + 1]
        if window_vals.size == 0:
            return 0.0
        hist, _ = np.histogram(window_vals, bins=15, range=(-5, 5), density=False)
        total = float(np.sum(hist))
        if total <= 0.0:
            return 0.0
        p_vec = hist / total
        H = -float(np.sum(p_vec * np.log(p_vec + eps)))
        return H / np.log(15.0)

    def _compute_log_likelihood(self, returns: np.ndarray) -> float:
        p = self.params
        if (
            np.any(~np.isfinite(self._mu))
            or np.any(~np.isfinite(self._E))
            or np.any(~np.isfinite(self._H))
            or np.any(np.abs(self._E) > p.E_max + 1e-9)
            or np.any(np.abs(self._mu) > p.max_mu + 1e-9)
            or np.any(np.abs(self._J) > p.J_max + 1e-9)
            or np.any((self._H < -1e-9) | (self._H > 1.0 + 1e-9))
        ):
            return -1e12

        sigma_safe = np.maximum(self._sigma, p.eps)
        z2 = ((returns - self._mu) ** 2) / (sigma_safe ** 2)
        ll_terms = -0.5 * (np.log(2.0 * np.pi * sigma_safe ** 2) + z2)
        if not np.all(np.isfinite(ll_terms)):
            return -1e12
        return float(np.sum(ll_terms))


# Minimal optimizer stub example -------------------------------------------
def single_pass_fit(returns: np.ndarray, params: Optional[Dict[str, float]] = None) -> TECDDriftModelV3:
    """Fit TECD-v3 once and return the model (helper stub for pipelines)."""
    m = TECDDriftModelV3(params=params)
    m.fit(returns)
    return m
