"""
Story 11.1: Calibrated Directional Confidence via Platt Scaling
================================================================

Transforms raw sign probabilities P(r > 0) into calibrated directional
confidence scores using Platt scaling (logistic calibration).

Key insight: raw model confidence != calibrated probability. If you say
conf=0.70, 70% of the time you should be correct. Platt scaling enforces
this via a logistic regression on logit-transformed raw probabilities.

Walk-forward calibration prevents future leakage: train on [t-500, t-1],
apply at t.

References:
    Platt (2000): "Probabilistic Outputs for Support Vector Machines"
    Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities With
    Supervised Learning"
"""
import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ===================================================================
# Constants
# ===================================================================

PLATT_TRAIN_WINDOW = 500       # Rolling training window size
PLATT_MIN_TRAIN = 100          # Minimum samples before calibration starts
PLATT_MAX_ITER = 100           # Max iterations for Platt fitting
PLATT_LR = 0.01               # Learning rate for gradient descent
PLATT_TOL = 1e-8               # Convergence tolerance
PLATT_CLIP_LOGIT = 10.0        # Clip logit to [-10, 10] for stability
PLATT_CLIP_PROB = 1e-7         # Clip raw probs away from 0 and 1
ECE_NUM_BINS = 10              # Number of bins for ECE calculation
RELIABILITY_TOLERANCE = 0.03   # Max deviation from diagonal in reliability


# ===================================================================
# Result Dataclass
# ===================================================================

@dataclass
class PlattCalibrationResult:
    """Result of Platt scaling calibration."""
    calibrated_probs: np.ndarray    # Calibrated probabilities at each time step
    a: float                        # Logistic slope parameter
    b: float                        # Logistic intercept parameter
    ece_before: float               # ECE before calibration
    ece_after: float                # ECE after calibration
    n_calibrated: int               # Number of time steps actually calibrated
    reliability_bins: np.ndarray    # Reliability diagram bin accuracies (10 bins)
    reliability_counts: np.ndarray  # Number of samples per bin


@dataclass
class WalkForwardCalibrationResult:
    """Result of walk-forward Platt calibration (no leakage)."""
    calibrated_probs: np.ndarray    # Calibrated probabilities for all time steps
    ece_before: float               # ECE before calibration (on calibrated subset)
    ece_after: float                # ECE after calibration (on calibrated subset)
    n_calibrated: int               # Number of time steps with calibration applied
    hit_rate_raw: float             # Hit rate using raw probs at conf > 0.60
    hit_rate_calibrated: float      # Hit rate using calibrated probs at conf > 0.60
    reliability_bins: np.ndarray    # Reliability diagram (10 bins)
    reliability_counts: np.ndarray  # Counts per bin
    a_final: float                  # Final Platt parameters
    b_final: float


# ===================================================================
# Core Functions
# ===================================================================

def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Compute logit(p) = log(p / (1-p)) with clipping for numerical stability."""
    p_clipped = np.clip(p, PLATT_CLIP_PROB, 1.0 - PLATT_CLIP_PROB)
    logit_val = np.log(p_clipped / (1.0 - p_clipped))
    return np.clip(logit_val, -PLATT_CLIP_LOGIT, PLATT_CLIP_LOGIT)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def _fit_platt(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Fit Platt scaling parameters (a, b) via maximum likelihood.

    Fits: p_cal = sigmoid(a * logit + b)
    Uses Newton's method with regularization for robustness.

    Args:
        logits: logit-transformed raw probabilities, shape (n,)
        labels: binary outcomes (1 = correct direction), shape (n,)

    Returns:
        (a, b) parameters
    """
    n = len(logits)
    if n < 2:
        return 1.0, 0.0

    # Initialize: a=1, b=0 (identity calibration)
    a, b = 1.0, 0.0

    for _ in range(PLATT_MAX_ITER):
        # Forward pass
        z = a * logits + b
        p = _sigmoid(z)

        # Gradient
        diff = p - labels
        grad_a = np.dot(diff, logits) / n
        grad_b = np.mean(diff)

        # Hessian diagonal (Newton step)
        w = p * (1.0 - p) + 1e-12
        H_aa = np.dot(w * logits, logits) / n + 1e-6
        H_bb = np.mean(w) + 1e-6

        # Newton update
        da = grad_a / H_aa
        db = grad_b / H_bb

        a -= da
        b -= db

        if abs(da) < PLATT_TOL and abs(db) < PLATT_TOL:
            break

    return float(a), float(b)


def platt_calibrate(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    validation_frac: float = 0.2,
) -> PlattCalibrationResult:
    """Fit Platt scaling on train split and evaluate on validation split.

    This is the simple (non-walk-forward) version for initial fitting and testing.

    Args:
        raw_probs: Raw P(r > 0) estimates, shape (n,)
        outcomes: Binary outcomes (1 = r > 0), shape (n,)
        validation_frac: Fraction of data for validation (from the end)

    Returns:
        PlattCalibrationResult
    """
    n = len(raw_probs)
    n_val = max(1, int(n * validation_frac))
    n_train = n - n_val

    # Split (temporal: train first, validate last)
    logits_train = _safe_logit(raw_probs[:n_train])
    labels_train = outcomes[:n_train].astype(float)
    logits_val = _safe_logit(raw_probs[n_train:])
    labels_val = outcomes[n_train:].astype(float)

    # Fit on train
    a, b = _fit_platt(logits_train, labels_train)

    # Apply to validation
    calibrated_val = _sigmoid(a * logits_val + b)

    # Apply to all
    logits_all = _safe_logit(raw_probs)
    calibrated_all = _sigmoid(a * logits_all + b)

    # ECE before (on validation)
    ece_before = compute_ece(raw_probs[n_train:], labels_val)
    ece_after = compute_ece(calibrated_val, labels_val)

    # Reliability diagram (on validation)
    bins, counts = compute_reliability_diagram(calibrated_val, labels_val)

    return PlattCalibrationResult(
        calibrated_probs=calibrated_all,
        a=a,
        b=b,
        ece_before=ece_before,
        ece_after=ece_after,
        n_calibrated=n_val,
        reliability_bins=bins,
        reliability_counts=counts,
    )


def platt_calibrate_walkforward(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    train_window: int = PLATT_TRAIN_WINDOW,
    min_train: int = PLATT_MIN_TRAIN,
) -> WalkForwardCalibrationResult:
    """Walk-forward Platt calibration with no future leakage.

    For each time step t, trains on [max(0, t-train_window) : t-1] and
    applies calibration at t.

    Args:
        raw_probs: Raw P(r > 0) estimates, shape (n,)
        outcomes: Binary outcomes (1 = r > 0), shape (n,)
        train_window: Rolling training window size
        min_train: Minimum training samples before calibration starts

    Returns:
        WalkForwardCalibrationResult
    """
    n = len(raw_probs)
    calibrated = raw_probs.copy().astype(float)
    n_calibrated = 0
    a_final, b_final = 1.0, 0.0

    logits_all = _safe_logit(raw_probs)

    for t in range(min_train, n):
        # Training window: [max(0, t - train_window), t)
        start = max(0, t - train_window)
        logits_train = logits_all[start:t]
        labels_train = outcomes[start:t].astype(float)

        # Fit Platt on training window
        a, b = _fit_platt(logits_train, labels_train)

        # Apply at time t
        calibrated[t] = float(_sigmoid(np.array([a * logits_all[t] + b]))[0])
        n_calibrated += 1
        a_final, b_final = a, b

    # Compute metrics on calibrated subset only
    cal_subset = calibrated[min_train:]
    raw_subset = raw_probs[min_train:]
    out_subset = outcomes[min_train:].astype(float)

    ece_before = compute_ece(raw_subset, out_subset)
    ece_after = compute_ece(cal_subset, out_subset)

    # Hit rate at conf > 0.60 threshold
    conf_raw = np.abs(2.0 * raw_subset - 1.0)
    conf_cal = np.abs(2.0 * cal_subset - 1.0)

    # For hit rate: predicted direction is sign(p - 0.5)
    pred_raw = (raw_subset > 0.5).astype(float)
    pred_cal = (cal_subset > 0.5).astype(float)

    mask_raw = conf_raw > 0.60
    mask_cal = conf_cal > 0.60

    if mask_raw.sum() > 0:
        hit_rate_raw = float(np.mean(pred_raw[mask_raw] == out_subset[mask_raw]))
    else:
        hit_rate_raw = 0.0

    if mask_cal.sum() > 0:
        hit_rate_cal = float(np.mean(pred_cal[mask_cal] == out_subset[mask_cal]))
    else:
        hit_rate_cal = 0.0

    # Reliability diagram on calibrated subset
    bins, counts = compute_reliability_diagram(cal_subset, out_subset)

    return WalkForwardCalibrationResult(
        calibrated_probs=calibrated,
        ece_before=ece_before,
        ece_after=ece_after,
        n_calibrated=n_calibrated,
        hit_rate_raw=hit_rate_raw,
        hit_rate_calibrated=hit_rate_cal,
        reliability_bins=bins,
        reliability_counts=counts,
        a_final=a_final,
        b_final=b_final,
    )


# ===================================================================
# Calibration Metrics
# ===================================================================

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_NUM_BINS,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where B_b is bin b, acc is observed accuracy, conf is mean predicted probability.

    Args:
        probs: Predicted probabilities, shape (n,)
        labels: Binary outcomes, shape (n,)
        n_bins: Number of equal-width bins

    Returns:
        ECE value (0 = perfectly calibrated)
    """
    n = len(probs)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (n_bin / n) * abs(acc - conf)

    return float(ece)


def compute_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_NUM_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability diagram bins.

    Returns mean observed frequency (accuracy) per bin and count per bin.

    Args:
        probs: Predicted probabilities, shape (n,)
        labels: Binary outcomes, shape (n,)
        n_bins: Number of bins

    Returns:
        (bin_accuracies, bin_counts) each shape (n_bins,)
        bin_accuracies[i] = NaN if bin is empty
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_bin = mask.sum()
        bin_counts[i] = n_bin
        if n_bin > 0:
            bin_accs[i] = labels[mask].mean()

    return bin_accs, bin_counts


def directional_confidence(prob_positive: float) -> float:
    """Convert P(r > 0) to directional confidence in [0, 1].

    conf_dir = |2 * P(r > 0) - 1|

    0.5 -> 0.0 (no confidence)
    0.8 -> 0.6 (moderate confidence)
    1.0 -> 1.0 (maximum confidence)

    Args:
        prob_positive: P(r > 0), must be in [0, 1]

    Returns:
        Directional confidence in [0, 1]
    """
    return abs(2.0 * prob_positive - 1.0)


def apply_platt_single(raw_prob: float, a: float, b: float) -> float:
    """Apply Platt scaling to a single raw probability.

    Args:
        raw_prob: Raw P(r > 0) in [0, 1]
        a: Platt slope
        b: Platt intercept

    Returns:
        Calibrated probability
    """
    logit_val = _safe_logit(np.array([raw_prob]))[0]
    z = a * logit_val + b
    return float(_sigmoid(np.array([z]))[0])
