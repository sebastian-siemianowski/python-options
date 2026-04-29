"""Shared numerical helpers for Student-t drift models.

The improved Student-t variants should differ in model structure, not in
duplicated calibration arithmetic.  Keep small distribution/statistic helpers
here so model classes stay focused on filtering and optimization.
"""

from __future__ import annotations

import math

import numpy as np


def clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
    """Clamp Student-t degrees of freedom with finite-input guards."""
    if not np.isfinite(nu):
        return float(nu_min)
    lo = max(2.001, float(nu_min))
    hi = max(lo, float(nu_max))
    return float(np.clip(float(nu), lo, hi))


def variance_to_scale(variance: float, nu: float) -> float:
    """Convert Student-t predictive variance to scale."""
    variance = float(variance) if np.isfinite(variance) else 1e-20
    variance = max(variance, 1e-20)
    nu = float(nu) if np.isfinite(nu) else 8.0
    if nu > 2.0:
        return float(math.sqrt(max(variance * (nu - 2.0) / nu, 1e-20)))
    return float(math.sqrt(variance))


def variance_to_scale_vec(variance: np.ndarray, nu: float) -> np.ndarray:
    """Vectorized Student-t variance-to-scale conversion."""
    variance = np.asarray(variance, dtype=np.float64)
    variance_safe = np.maximum(np.where(np.isfinite(variance), variance, 1e-20), 1e-20)
    nu = float(nu) if np.isfinite(nu) else 8.0
    if nu > 2.0:
        scale = np.sqrt(variance_safe * (nu - 2.0) / nu)
    else:
        scale = np.sqrt(variance_safe)
    return np.maximum(scale, 1e-10)


def precompute_vov(vol: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling std of log-volatility with robust finite guards."""
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(vol)
    if n <= 0:
        return np.empty(0, dtype=np.float64)

    finite = vol[np.isfinite(vol) & (vol > 0)]
    fill = float(np.median(finite)) if finite.size else 1.0
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, fill)
    vol = np.maximum(vol, max(fill * 1e-4, 1e-10))

    window = int(max(2, min(window, max(n, 2))))
    if n <= window:
        return np.zeros(n, dtype=np.float64)

    log_vol = np.log(vol)
    cs1 = np.concatenate(([0.0], np.cumsum(log_vol)))
    cs2 = np.concatenate(([0.0], np.cumsum(log_vol * log_vol)))
    inv_w = 1.0 / float(window)
    idx = np.arange(window, n)
    s1 = cs1[idx] - cs1[idx - window]
    s2 = cs2[idx] - cs2[idx - window]
    var_arr = np.maximum(s2 * inv_w - (s1 * inv_w) ** 2, 0.0)
    vov = np.empty(n, dtype=np.float64)
    vov[window:] = np.sqrt(var_arr)
    vov[:window] = vov[window] if window < n else 0.0
    return np.where(np.isfinite(vov), vov, 0.0)


def compute_cvm_statistic(pit_values: np.ndarray) -> float:
    """Cramer-von Mises W2 for Uniform(0,1) PIT values."""
    pit_values = np.asarray(pit_values, dtype=np.float64).ravel()
    pit_values = pit_values[np.isfinite(pit_values)]
    n = len(pit_values)
    if n < 2:
        return float("inf")
    u = np.sort(np.clip(pit_values, 1e-12, 1.0 - 1e-12))
    i_vals = np.arange(1, n + 1, dtype=np.float64)
    w2 = float(np.sum((u - (2.0 * i_vals - 1.0) / (2.0 * n)) ** 2) + 1.0 / (12.0 * n))
    return w2 if np.isfinite(w2) else float("inf")


def compute_ad_statistic(pit_values: np.ndarray) -> float:
    """Anderson-Darling A2 for Uniform(0,1) PIT values."""
    pit_values = np.asarray(pit_values, dtype=np.float64).ravel()
    pit_values = pit_values[np.isfinite(pit_values)]
    n = len(pit_values)
    if n < 2:
        return float("inf")
    u = np.sort(np.clip(pit_values, 1e-10, 1.0 - 1e-10))
    i_vals = np.arange(1, n + 1, dtype=np.float64)
    a2 = -float(n) - float(np.sum((2.0 * i_vals - 1.0) * (np.log(u) + np.log1p(-u[::-1])))) / float(n)
    return float(max(a2, 0.0)) if np.isfinite(a2) else float("inf")


def ewm_lagged_correction(returns: np.ndarray, mu_pred: np.ndarray, ewm_lambda: float) -> np.ndarray:
    """Causal EWM correction from lagged forecast innovations."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    mu_pred = np.asarray(mu_pred, dtype=np.float64).ravel()
    n = min(len(returns), len(mu_pred))
    corrections = np.zeros(n, dtype=np.float64)
    if n <= 2 or ewm_lambda < 0.01:
        return corrections

    lam = float(np.clip(ewm_lambda, 0.01, 0.999))
    alpha = 1.0 - lam
    innov_lagged = np.asarray(returns[:n - 1] - mu_pred[:n - 1], dtype=np.float64)
    innov_lagged = np.where(np.isfinite(innov_lagged), innov_lagged, 0.0)

    if n <= 4096:
        value = 0.0
        for idx in range(n - 1):
            value = lam * value + alpha * innov_lagged[idx]
            corrections[idx + 1] = value
        return corrections

    try:
        from scipy.signal import lfilter

        corrections[1:] = lfilter([alpha], [1.0, -lam], innov_lagged)
    except Exception:
        value = 0.0
        for idx in range(n - 1):
            value = lam * value + alpha * innov_lagged[idx]
            corrections[idx + 1] = value
    return corrections


def pit_simple_path(
    returns_test: np.ndarray,
    mu_pred_test: np.ndarray,
    s_calibrated: np.ndarray,
    nu: float,
    t_df_asym: float,
    cdf_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basic Student-t PIT path shared by the improved Student-t variants."""
    returns_test = np.asarray(returns_test, dtype=np.float64).ravel()
    mu_pred_test = np.asarray(mu_pred_test, dtype=np.float64).ravel()
    s_calibrated = np.asarray(s_calibrated, dtype=np.float64).ravel()
    n = min(len(returns_test), len(mu_pred_test), len(s_calibrated))
    if n <= 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    nu = float(nu) if np.isfinite(nu) else 8.0
    t_df_asym = float(t_df_asym) if np.isfinite(t_df_asym) else 0.0
    returns_test = returns_test[:n]
    mu_pred_test = mu_pred_test[:n]
    s_calibrated = s_calibrated[:n]

    sigma = variance_to_scale_vec(s_calibrated, nu)
    z = (returns_test - mu_pred_test) / sigma
    z = np.where(np.isfinite(z), z, 0.0)

    if cdf_fn is None:
        from scipy.stats import t as student_t

        def cdf_fn(values, df):
            return student_t.cdf(values, df=float(df))

    if abs(t_df_asym) > 0.05:
        pit_values = np.zeros(n, dtype=np.float64)
        nu_l = max(2.5, nu - t_df_asym)
        nu_r = max(2.5, nu + t_df_asym)
        left = z < 0.0
        pit_values[left] = cdf_fn(z[left], nu_l)
        pit_values[~left] = cdf_fn(z[~left], nu_r)
    else:
        pit_values = cdf_fn(z, nu)

    pit_values = np.asarray(pit_values, dtype=np.float64)
    pit_values = np.where(np.isfinite(pit_values), pit_values, 0.5)
    return np.clip(pit_values, 0.001, 0.999), sigma, mu_pred_test


def ks_uniform_approx(pit_values: np.ndarray) -> tuple[float, float]:
    """Fast Kolmogorov-Smirnov approximation against Uniform(0,1)."""
    pit_values = np.asarray(pit_values, dtype=np.float64).ravel()
    pit_values = pit_values[np.isfinite(pit_values)]
    n = len(pit_values)
    if n < 2:
        return 1.0, 0.0
    sorted_pit = np.sort(np.clip(pit_values, 0.0, 1.0))
    ecdf = np.arange(1, n + 1, dtype=np.float64) / float(n)
    d_plus = float(np.max(ecdf - sorted_pit))
    d_minus = float(np.max(sorted_pit - np.arange(0, n, dtype=np.float64) / float(n)))
    d_stat = max(d_plus, d_minus)
    sqrt_n = math.sqrt(float(n))
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_stat
    if lam < 0.001:
        return d_stat, 1.0
    if lam > 3.0:
        return d_stat, 0.0
    lam2 = lam * lam
    p_value = 2.0 * (
        math.exp(-2.0 * lam2)
        - math.exp(-8.0 * lam2)
        + math.exp(-18.0 * lam2)
        - math.exp(-32.0 * lam2)
    )
    return d_stat, float(np.clip(p_value, 0.0, 1.0))


def ad_correction_pipeline_student_t(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    scale: np.ndarray,
    nu: float,
    pit_raw: np.ndarray,
    *,
    min_obs: int = 250,
    cdf_fn=None,
    ks_fn=None,
    ad_fn=None,
    nu_min: float = 2.1,
    nu_max: float = 60.0,
    gpd_method: str = "pwm",
) -> tuple[np.ndarray, dict]:
    """Shared AD/TWSC/SPTG/isotonic transport for Student-t PIT calibration."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    mu_pred = np.asarray(mu_pred, dtype=np.float64).ravel()
    scale = np.asarray(scale, dtype=np.float64).ravel()
    pit = np.asarray(pit_raw, dtype=np.float64).ravel()
    n = min(len(returns), len(mu_pred), len(scale), len(pit))
    empty_diag = {
        "twsc_applied": False,
        "sptg_applied": False,
        "isotonic_applied": False,
        "calibration_params": {},
        "n_obs": int(max(n, 0)),
    }
    if n <= 0:
        return np.empty(0, dtype=np.float64), empty_diag

    returns = returns[:n].copy()
    mu_pred = mu_pred[:n].copy()
    scale = scale[:n].copy()
    pit = pit[:n].copy()

    returns[~np.isfinite(returns)] = 0.0
    mu_pred[~np.isfinite(mu_pred)] = 0.0
    finite_scale = scale[np.isfinite(scale) & (scale > 0.0)]
    scale_fill = float(np.median(finite_scale)) if finite_scale.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, scale_fill)
    scale = np.maximum(scale, max(scale_fill * 1e-6, 1e-10))
    pit = np.clip(np.where(np.isfinite(pit), pit, 0.5), 0.001, 0.999)

    if n < 50:
        return pit, empty_diag

    if cdf_fn is None:
        from scipy.stats import t as student_t

        def cdf_fn(values, df):
            return student_t.cdf(values, df=float(df))

    ks_fn = ks_fn or ks_uniform_approx
    ad_fn = ad_fn or compute_ad_statistic
    nu = clip_nu(nu, nu_min, nu_max)
    diag = {
        "twsc_applied": False,
        "sptg_applied": False,
        "sptg_xi_left": float("nan"),
        "sptg_xi_right": float("nan"),
        "isotonic_applied": False,
        "isotonic_ks_improvement": 0.0,
        "n_obs": int(n),
    }
    cal_params: dict[str, float | list[float]] = {}

    z = np.clip((returns - mu_pred) / np.maximum(scale, 1e-10), -1e6, 1e6)
    pit_base = pit.copy()
    _ks_base, p_base = ks_fn(pit_base)
    ad_base = ad_fn(pit_base)
    z_for_gpd = z

    scale_inflate = None
    try:
        from models.numba_wrappers import run_ad_twsc

        scale_inflate = run_ad_twsc(
            z,
            ewma_lambda=0.97,
            alpha_quantile=0.05,
            kappa=0.5,
            max_inflate=2.0,
            deadzone=0.15,
        )
    except (ImportError, Exception):
        try:
            from models.numba_kernels import ad_twsc_kernel

            scale_inflate = ad_twsc_kernel(
                np.ascontiguousarray(z, dtype=np.float64),
                0.97,
                0.05,
                0.5,
                2.0,
                0.15,
            )
        except Exception:
            scale_inflate = None

    if scale_inflate is not None:
        scale_inflate = np.asarray(scale_inflate, dtype=np.float64).ravel()
        if len(scale_inflate) >= n:
            scale_inflate = np.clip(
                np.where(np.isfinite(scale_inflate[:n]), scale_inflate[:n], 1.0),
                1.0,
                2.5,
            )
            z_twsc = z / scale_inflate
            pit_twsc = np.clip(cdf_fn(z_twsc, nu), 0.001, 0.999)
            _ks_twsc, p_twsc = ks_fn(pit_twsc)
            ad_twsc = ad_fn(pit_twsc)
            if (p_twsc >= p_base * 0.95) or (ad_twsc <= ad_base * 1.02):
                pit = pit_twsc
                diag["twsc_applied"] = True
                tail_start = max(1, int(n * 0.7))
                tail_factors = scale_inflate[tail_start:]
                tail_factors = tail_factors[tail_factors > 0.0]
                if len(tail_factors) > 0:
                    twsc_geo_mean = float(np.exp(np.mean(np.log(tail_factors))))
                    cal_params["twsc_scale_factor"] = float(np.clip(twsc_geo_mean, 1.0, 2.5))
                    cal_params["twsc_last_ewma"] = float(scale_inflate[-1])
                z_for_gpd = z_twsc
                p_base, ad_base = p_twsc, ad_twsc

    if n >= min_obs:
        try:
            from calibration.evt_tail import fit_gpd_pot

            z_for_gpd = np.asarray(z_for_gpd, dtype=np.float64)
            abs_z = np.abs(z_for_gpd)
            left_losses = abs_z[z_for_gpd < 0.0]
            right_losses = abs_z[z_for_gpd > 0.0]

            if len(left_losses) >= 25 and len(right_losses) >= 25:
                gpd_left = fit_gpd_pot(left_losses, threshold_percentile=0.90, method=gpd_method)
                gpd_right = fit_gpd_pot(right_losses, threshold_percentile=0.90, method=gpd_method)
            else:
                gpd_left = None
                gpd_right = None

            if gpd_left is not None and gpd_right is not None and gpd_left.fit_success and gpd_right.fit_success:
                u_left = float(gpd_left.threshold)
                u_right = float(gpd_right.threshold)
                p_left_val = float(cdf_fn(np.array([-u_left], dtype=np.float64), nu)[0])
                p_right_val = float(1.0 - cdf_fn(np.array([u_right], dtype=np.float64), nu)[0])

                if p_left_val > 0.001 and p_right_val > 0.001 and u_left > 0.5 and u_right > 0.5:
                    try:
                        from models.numba_wrappers import run_ad_sptg_student_t

                        pit_sptg = run_ad_sptg_student_t(
                            z_for_gpd,
                            nu,
                            gpd_left.xi,
                            gpd_left.sigma,
                            u_left,
                            gpd_right.xi,
                            gpd_right.sigma,
                            u_right,
                            p_left_val,
                            p_right_val,
                        )
                    except (ImportError, Exception):
                        from models.numba_kernels import ad_sptg_cdf_student_t_array

                        pit_sptg = ad_sptg_cdf_student_t_array(
                            np.ascontiguousarray(z_for_gpd, dtype=np.float64),
                            nu,
                            gpd_left.xi,
                            gpd_left.sigma,
                            u_left,
                            gpd_right.xi,
                            gpd_right.sigma,
                            u_right,
                            p_left_val,
                            p_right_val,
                        )
                    pit_sptg = np.clip(np.asarray(pit_sptg, dtype=np.float64), 0.001, 0.999)
                    _ks_sptg, p_sptg = ks_fn(pit_sptg)
                    ad_sptg = ad_fn(pit_sptg)
                    if (p_sptg >= p_base * 0.95) or (ad_sptg <= ad_base):
                        pit = pit_sptg
                        diag["sptg_applied"] = True
                        diag["sptg_xi_left"] = float(gpd_left.xi)
                        diag["sptg_xi_right"] = float(gpd_right.xi)
                        xi_max = max(abs(float(gpd_left.xi)), abs(float(gpd_right.xi)))
                        if xi_max > 0.02:
                            nu_from_gpd = 1.0 / xi_max
                            nu_effective = max(2.5, min(nu, nu_from_gpd))
                            cal_params["nu_effective"] = float(nu_effective)
                            cal_params["nu_adjustment_ratio"] = float(nu_effective / nu)
                        else:
                            cal_params["nu_effective"] = float(nu)
                            cal_params["nu_adjustment_ratio"] = 1.0
                        p_base, ad_base = p_sptg, ad_sptg
        except Exception:
            pass

    if n >= 100:
        try:
            from calibration.isotonic_recalibration import IsotonicRecalibrator

            recal = IsotonicRecalibrator()
            result = recal.fit(pit)
            if result.fit_success and not result.is_identity:
                pit_iso = np.clip(recal.transform(pit), 0.001, 0.999)
                _ks_before, p_before = ks_fn(pit)
                _ks_after, p_after = ks_fn(pit_iso)
                ad_before = ad_fn(pit)
                ad_after = ad_fn(pit_iso)
                if (p_after >= p_before) or (ad_after <= ad_before):
                    pit = pit_iso
                    diag["isotonic_applied"] = True
                    diag["isotonic_ks_improvement"] = float(p_after - p_before)
                    cal_params["isotonic_x_knots"] = result.x_knots.tolist()
                    cal_params["isotonic_y_knots"] = result.y_knots.tolist()
        except Exception:
            pass

    diag["calibration_params"] = cal_params
    return np.clip(pit, 0.001, 0.999), diag


def compute_berkowitz_full(pit_values: np.ndarray) -> tuple[float, float, int]:
    """
    Berkowitz LR test: H0 Phi^-1(PIT)~N(0,1) iid vs H1 AR(1). Chi2(3).

    Returns (p_value, lr_statistic, n_pit).
    """
    try:
        from scipy.special import chdtrc, ndtri

        pit = np.asarray(pit_values, dtype=np.float64).ravel()
        pit = pit[np.isfinite(pit)]
        z = ndtri(np.clip(pit, 1e-6, 1.0 - 1e-6))
        z = z[np.isfinite(z)]
        n_z = len(z)
        if n_z <= 20:
            return (float("nan"), 0.0, n_z)

        mu_hat = float(np.mean(z))
        var_hat = max(float(np.var(z, ddof=0)), 1e-6)
        z_c = z - mu_hat
        denom = float(np.sum(z_c[:-1] ** 2))
        rho_hat = 0.0
        if denom > 1e-12:
            rho_hat = float(np.clip(np.sum(z_c[1:] * z_c[:-1]) / denom, -0.99, 0.99))

        ll_null = -0.5 * n_z * math.log(2.0 * math.pi) - 0.5 * float(np.sum(z ** 2))
        sigma_sq_cond = max(var_hat * (1.0 - rho_hat ** 2), 1e-6)
        resid = z[1:] - (mu_hat + rho_hat * (z[:-1] - mu_hat))
        ll_alt = (
            -0.5 * math.log(2.0 * math.pi * var_hat)
            -0.5 * (z[0] - mu_hat) ** 2 / var_hat
            -0.5 * (n_z - 1) * math.log(2.0 * math.pi * sigma_sq_cond)
            -0.5 * float(np.sum(resid ** 2)) / sigma_sq_cond
        )
        lr_stat = float(max(2.0 * (ll_alt - ll_null), 0.0))
        p_value = float(chdtrc(3, lr_stat))
        return (p_value if np.isfinite(p_value) else float("nan"), lr_stat, n_z)
    except Exception:
        return (float("nan"), 0.0, 0)


def compute_berkowitz_pvalue(pit_values: np.ndarray) -> float:
    """Berkowitz p-value convenience wrapper."""
    return compute_berkowitz_full(pit_values)[0]


def estimate_gjr_garch_params(
    returns_train: np.ndarray,
    mu_pred_train: np.ndarray,
    mu_drift_opt: float,
    n_train: int,
    *,
    alpha_default: float = 0.06,
    beta_anchor: float = 0.96,
    alpha_bounds: tuple[float, float] = (0.02, 0.20),
    beta_bounds: tuple[float, float] = (0.65, 0.94),
    leverage_cap: float = 0.18,
    persistence_cap: float = 0.985,
    short_beta: float = 0.88,
) -> dict[str, float]:
    """
    Robust moment estimator for GJR-GARCH(1,1) parameters.

    Uses winsorized residual moments and explicit stationarity enforcement so a
    single bad print cannot dominate volatility blending.
    """
    returns_train = np.asarray(returns_train, dtype=np.float64).ravel()
    mu_pred_train = np.asarray(mu_pred_train, dtype=np.float64).ravel()
    n = min(len(returns_train), len(mu_pred_train), int(max(n_train, 0)))
    if n <= 2:
        return {
            "garch_omega": 1e-10,
            "garch_alpha": float(alpha_default),
            "garch_beta": 0.90,
            "garch_leverage": 0.0,
            "unconditional_var": 1e-8,
        }

    innovations = returns_train[:n] - mu_pred_train[:n] - float(mu_drift_opt)
    innovations = innovations[np.isfinite(innovations)]
    if len(innovations) <= 2:
        innovations = np.zeros(3, dtype=np.float64)

    med = float(np.median(innovations))
    mad = float(np.median(np.abs(innovations - med)))
    robust_scale = max(1.4826 * mad, float(np.std(innovations)), 1e-8)
    cap = max(8.0 * robust_scale, 1e-8)
    innovations_clip = np.clip(innovations - med, -cap, cap)
    sq_innov = innovations_clip ** 2
    unconditional_var = max(float(np.mean(sq_innov)), 1e-10)

    garch_leverage = 0.0
    if len(innovations_clip) > 100:
        sq_centered = sq_innov - unconditional_var
        denom = float(np.sum(sq_centered[:-1] ** 2))
        if denom > 1e-12:
            garch_alpha = float(np.sum(sq_centered[1:] * sq_centered[:-1]) / denom)
            garch_alpha = float(np.clip(garch_alpha, alpha_bounds[0], alpha_bounds[1]))
        else:
            garch_alpha = float(alpha_default)

        neg_indicator = (innovations_clip[:-1] < 0.0).astype(np.float64)
        n_neg = max(int(np.sum(neg_indicator)), 1)
        n_pos = max(int(np.sum(1.0 - neg_indicator)), 1)
        mean_sq_after_neg = float(np.sum(sq_innov[1:] * neg_indicator) / n_neg)
        mean_sq_after_pos = float(np.sum(sq_innov[1:] * (1.0 - neg_indicator)) / n_pos)
        leverage_ratio = mean_sq_after_neg / max(mean_sq_after_pos, 1e-12)
        if leverage_ratio > 1.0:
            garch_leverage = float(np.clip(garch_alpha * (leverage_ratio - 1.0), 0.0, leverage_cap))

        garch_beta = beta_anchor - garch_alpha - 0.5 * garch_leverage
        garch_beta = float(np.clip(garch_beta, beta_bounds[0], beta_bounds[1]))
        persistence = garch_alpha + 0.5 * garch_leverage + garch_beta
        if persistence >= persistence_cap:
            garch_beta = max(0.50, persistence_cap - garch_alpha - 0.5 * garch_leverage)

        persistence = min(garch_alpha + 0.5 * garch_leverage + garch_beta, persistence_cap)
        garch_omega = max(unconditional_var * (1.0 - persistence), 1e-10)
    else:
        garch_alpha = float(alpha_default)
        garch_beta = float(short_beta)
        garch_omega = max(unconditional_var * (1.0 - garch_alpha - garch_beta), 1e-10)

    return {
        "garch_omega": float(garch_omega),
        "garch_alpha": float(garch_alpha),
        "garch_beta": float(garch_beta),
        "garch_leverage": float(garch_leverage),
        "unconditional_var": float(unconditional_var),
    }


_ASSET_PHI_CENTER = {
    "index": 0.80,
    "large_cap": 0.65,
    "small_cap": 0.20,
    "high_vol_equity": 0.05,
    "crypto": 0.45,
    "forex": 0.30,
    "metals_gold": 0.75,
    "metals_silver": 0.55,
    "metals_other": 0.45,
}

_ASSET_PHI_TAU = {
    "index": 0.80,
    "large_cap": 0.75,
    "small_cap": 0.60,
    "high_vol_equity": 0.50,
    "crypto": 0.75,
    "forex": 0.65,
    "metals_gold": 0.80,
    "metals_silver": 0.70,
    "metals_other": 0.70,
}

_ASSET_PHI_STRENGTH = {
    "index": 1.00,
    "large_cap": 0.85,
    "small_cap": 0.55,
    "high_vol_equity": 0.35,
    "crypto": 0.65,
    "forex": 0.50,
    "metals_gold": 0.85,
    "metals_silver": 0.70,
    "metals_other": 0.65,
}


def asset_phi_profile(asset_class: str | None) -> tuple[float, float, float]:
    """Return (center, tau, strength) for weak asset-aware phi regularization."""
    return (
        float(_ASSET_PHI_CENTER.get(asset_class, 0.25)),
        float(_ASSET_PHI_TAU.get(asset_class, 0.70)),
        float(_ASSET_PHI_STRENGTH.get(asset_class, 0.60)),
    )
