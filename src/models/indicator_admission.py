"""Admission gates for indicator-integrated model candidates.

The functions here do not tune or trade.  They encode the guardrails that an
indicator model must pass before it is allowed to compete beside its no-indicator
control: registered feature contract, explicit control, BIC support, optional
LFO support, and low-redundancy feature transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from models.indicator_state import channels_for_specs, get_indicator_feature_spec
from models.model_registry import (
    ModelSpec,
    create_indicator_integrated_spec,
    get_model_spec,
)


CHANNEL_PARAM_NAMES = {
    "mean": "ind_mean_weight",
    "variance": "ind_variance_weight",
    "tail": "ind_tail_weight",
    "asymmetry": "ind_asymmetry_weight",
    "q": "ind_q_weight",
    "regime": "ind_regime_weight",
    "calibration": "ind_calibration_weight",
    "confidence": "ind_confidence_weight",
}


@dataclass(frozen=True)
class IndicatorAdmissionDecision:
    """Auditable decision for one indicator candidate versus its control."""

    candidate_name: str
    control_name: str
    accepted: bool
    # Rectification convention: candidate - control.  Negative BIC delta is
    # better because lower/more negative BIC wins in benchmark artifacts.
    bic_delta: float
    lfo_delta: Optional[float]
    reason: str


def indicator_param_names_for_features(indicator_features: Sequence[str]) -> Tuple[str, ...]:
    """Return canonical extra parameter names implied by feature channels."""
    for name in indicator_features:
        if get_indicator_feature_spec(name) is None:
            raise KeyError(f"unknown indicator feature spec: {name}")
    return tuple(CHANNEL_PARAM_NAMES[ch] for ch in channels_for_specs(indicator_features))


def create_indicator_candidate_family(
    base_model_names: Sequence[str],
    indicator_key: str,
    indicator_features: Tuple[str, ...],
    extra_param_names: Optional[Tuple[str, ...]] = None,
) -> Tuple[ModelSpec, ...]:
    """Create side-by-side candidate specs from registered no-indicator controls."""
    if extra_param_names is None:
        extra_param_names = indicator_param_names_for_features(indicator_features)
    specs = []
    for base_name in base_model_names:
        base_spec = get_model_spec(base_name)
        if base_spec is None:
            raise KeyError(f"unknown base model: {base_name}")
        if base_spec.is_indicator_integrated:
            raise ValueError("candidate families must be built from controls")
        specs.append(
            create_indicator_integrated_spec(
                base_name,
                indicator_key,
                indicator_features,
                extra_param_names=extra_param_names,
            )
        )
    return tuple(specs)


def decide_indicator_admission(
    candidate_name: str,
    control_name: str,
    candidate_bic: float,
    control_bic: float,
    candidate_lfo: Optional[float] = None,
    control_lfo: Optional[float] = None,
    max_bic_delta: float = 0.0,
    min_bic_delta: Optional[float] = None,
    min_lfo_delta: float = 0.0,
) -> IndicatorAdmissionDecision:
    """Decide if a candidate beats its control under BIC and optional LFO gates."""
    if not np.isfinite(candidate_bic) or not np.isfinite(control_bic):
        return IndicatorAdmissionDecision(candidate_name, control_name, False, float("nan"), None, "non-finite BIC")
    if min_bic_delta is not None:
        max_bic_delta = -float(min_bic_delta)
    bic_delta = float(candidate_bic - control_bic)
    if bic_delta > max_bic_delta:
        return IndicatorAdmissionDecision(
            candidate_name, control_name, False, bic_delta, None, "BIC gate failed"
        )
    lfo_delta = None
    if candidate_lfo is not None or control_lfo is not None:
        if candidate_lfo is None or control_lfo is None:
            return IndicatorAdmissionDecision(
                candidate_name, control_name, False, bic_delta, None, "incomplete LFO gate"
            )
        if not np.isfinite(candidate_lfo) or not np.isfinite(control_lfo):
            return IndicatorAdmissionDecision(
                candidate_name, control_name, False, bic_delta, None, "non-finite LFO"
            )
        lfo_delta = float(candidate_lfo - control_lfo)
        if lfo_delta < min_lfo_delta:
            return IndicatorAdmissionDecision(
                candidate_name, control_name, False, bic_delta, lfo_delta, "LFO gate failed"
            )
    return IndicatorAdmissionDecision(candidate_name, control_name, True, bic_delta, lfo_delta, "accepted")


def assert_candidate_controls_present(
    candidate_specs: Iterable[ModelSpec],
    available_model_names: Iterable[str],
) -> None:
    """Fail closed when an indicator candidate does not have its control present."""
    available = set(available_model_names)
    missing = {
        spec.name: spec.base_model_name
        for spec in candidate_specs
        if spec.is_indicator_integrated and spec.base_model_name not in available
    }
    if missing:
        detail = ", ".join(f"{name}->{control}" for name, control in sorted(missing.items()))
        raise AssertionError(f"indicator candidates missing controls: {detail}")


def prune_correlated_indicator_columns(
    feature_matrix: Sequence[Sequence[float]],
    output_names: Sequence[str],
    max_abs_corr: float = 0.97,
) -> Tuple[str, ...]:
    """Keep a low-redundancy ordered subset of indicator feature columns."""
    if not 0.0 < max_abs_corr < 1.0:
        raise ValueError("max_abs_corr must be in (0, 1)")
    x = np.asarray(feature_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    if x.shape[1] != len(output_names):
        raise ValueError("output_names length must match feature columns")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    kept_idx = []
    for j in range(x.shape[1]):
        col = x[:, j]
        col_std = float(np.std(col))
        if col_std <= 1e-12:
            continue
        keep = True
        for k in kept_idx:
            prev = x[:, k]
            prev_std = max(float(np.std(prev)), 1e-12)
            corr = float(np.mean((col - np.mean(col)) * (prev - np.mean(prev))) / (col_std * prev_std))
            if abs(corr) >= max_abs_corr:
                keep = False
                break
        if keep:
            kept_idx.append(j)
    return tuple(output_names[j] for j in kept_idx)


def summarize_admission_decisions(
    decisions: Sequence[IndicatorAdmissionDecision],
) -> Mapping[str, int]:
    """Return compact accepted/rejected counts for audit output."""
    accepted = sum(1 for decision in decisions if decision.accepted)
    return {
        "accepted": accepted,
        "rejected": len(decisions) - accepted,
        "total": len(decisions),
    }
