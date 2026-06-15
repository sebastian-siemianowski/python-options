"""
Guardrails for politician disclosure context.

Politician trading activity is intentionally contextual research in the MVP.
It must not enter production BMA weights, PIT calibration, Kelly sizing, or
high-conviction labels without a separate no-leakage research gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, sqrt, tanh
from typing import Final, Iterable


POLITICIAN_ACTIVITY_CLASSIFICATION: Final[str] = "contextual_research"
PROHIBITED_CLASSIFICATION: Final[str] = "model_feature"
DEFAULT_EVENT_TIME_ANCHOR: Final[str] = "disclosure_date"
ALLOWED_EVENT_TIME_ANCHORS: Final[tuple[str, ...]] = ("disclosure_date",)

ALLOWED_CONTEXT_SURFACES: Final[tuple[str, ...]] = (
    "politicians_page",
    "watchlist_badge",
    "signal_table_context_column",
    "chart_timeline_overlay",
    "research_report",
)

PROHIBITED_MODEL_INTEGRATION_TARGETS: Final[tuple[str, ...]] = (
    "bma_weights",
    "pit_calibration",
    "kelly_sizing",
    "high_conviction_labels",
    "latest_signals_output",
)


@dataclass(frozen=True)
class PoliticianContextPolicy:
    """Machine-readable policy for safe politician context usage."""

    classification: str = POLITICIAN_ACTIVITY_CLASSIFICATION
    default_event_time_anchor: str = DEFAULT_EVENT_TIME_ANCHOR
    allowed_surfaces: tuple[str, ...] = ALLOWED_CONTEXT_SURFACES
    prohibited_targets: tuple[str, ...] = PROHIBITED_MODEL_INTEGRATION_TARGETS


def get_politician_context_policy() -> PoliticianContextPolicy:
    """Return the current production policy for politician disclosure context."""
    return PoliticianContextPolicy()


def assert_contextual_research_only(classification: str) -> None:
    """Reject attempts to classify politician activity as a production feature."""
    if classification != POLITICIAN_ACTIVITY_CLASSIFICATION:
        raise ValueError(
            "Politician activity is contextual_research only in MVP; "
            f"received classification={classification!r}."
        )


def assert_not_model_integration_target(target: str) -> None:
    """Reject use in production model, calibration, sizing, or labels."""
    normalized = target.strip().lower()
    if normalized in PROHIBITED_MODEL_INTEGRATION_TARGETS:
        raise ValueError(
            f"Politician context cannot feed {target!r} in MVP. "
            "Run a separate no-leakage research gate first."
        )


def validate_no_leakage_research_gate(
    *,
    event_time_anchor: str,
    requested_targets: Iterable[str] = (),
) -> dict[str, object]:
    """
    Validate a future research integration gate.

    The only allowed event-time anchor is disclosure_date because it represents
    when the information became public enough for the system to know it.
    """
    anchor = event_time_anchor.strip().lower()
    if anchor not in ALLOWED_EVENT_TIME_ANCHORS:
        raise ValueError(
            "Politician research gates must use disclosure_date to avoid "
            f"lookahead leakage; received {event_time_anchor!r}."
        )

    blocked_targets = [
        target
        for target in requested_targets
        if target.strip().lower() in PROHIBITED_MODEL_INTEGRATION_TARGETS
    ]
    if blocked_targets:
        raise ValueError(
            "Research gate cannot directly approve production integration "
            f"targets without a separate promotion review: {blocked_targets}."
        )

    return {
        "classification": POLITICIAN_ACTIVITY_CLASSIFICATION,
        "event_time_anchor": anchor,
        "approved_for_research": True,
        "approved_for_model_feature": False,
    }


def compute_politician_activity_score(
    trades: list[dict],
    *,
    as_of_date: str | None = None,
) -> dict[str, object]:
    """Compute bounded contextual activity score from public disclosure rows."""
    if not trades:
        return _empty_activity_score()
    as_of = date.fromisoformat(as_of_date) if as_of_date else date.today()
    weighted_net = 0.0
    weighted_abs = 0.0
    confidences = []
    filers = set()
    recency_weights = []
    for row in trades:
        direction = _transaction_direction(row.get("transaction_type"))
        amount = float(row.get("amount_mid_usd") or row.get("amount_min_usd") or 1.0)
        disclosure = row.get("disclosure_date")
        recency = _recency_decay(disclosure, as_of)
        parser_conf = float(row.get("parser_confidence") or 0.5)
        weight = max(1.0, amount) ** 0.5 * recency * parser_conf
        weighted_net += direction * weight
        weighted_abs += abs(weight)
        confidences.append(parser_conf)
        recency_weights.append(recency)
        if row.get("filer_name"):
            filers.add(str(row["filer_name"]))
    raw = weighted_net / weighted_abs if weighted_abs else 0.0
    score = tanh(raw)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    unique_filer_factor = min(1.0, sqrt(max(1, len(filers)) / 5.0))
    sample_factor = min(1.0, len(trades) / 3.0)
    confidence = round(max(0.0, min(1.0, avg_conf * unique_filer_factor * sample_factor)), 3)
    return {
        "politician_activity_score": round(score, 3),
        "confidence": confidence,
        "explanation": {
            "classification": POLITICIAN_ACTIVITY_CLASSIFICATION,
            "components": {
                "buy_sell_imbalance": round(raw, 3),
                "amount_midpoint_weighting": "sqrt(amount_mid_usd or amount_min_usd)",
                "average_recency_decay": round(sum(recency_weights) / len(recency_weights), 3),
                "unique_filers": len(filers),
                "average_parser_confidence": round(avg_conf, 3),
            },
            "bounded_range": "[-1, 1]",
            "positive_means": "net disclosed purchases",
            "negative_means": "net disclosed sales",
            "model_usage": "contextual_research_only_not_bma_or_kelly",
        },
    }


def _empty_activity_score() -> dict[str, object]:
    return {
        "politician_activity_score": 0.0,
        "confidence": 0.0,
        "explanation": {
            "classification": POLITICIAN_ACTIVITY_CLASSIFICATION,
            "components": {},
            "bounded_range": "[-1, 1]",
            "model_usage": "contextual_research_only_not_bma_or_kelly",
        },
    }


def _transaction_direction(value) -> float:
    normalized = str(value or "").lower()
    if normalized in {"purchase", "received"}:
        return 1.0
    if normalized in {"sale", "sale_partial"}:
        return -1.0
    return 0.0


def _recency_decay(disclosure_date, as_of: date) -> float:
    try:
        event_date = date.fromisoformat(str(disclosure_date))
    except Exception:
        return 0.25
    days = max(0, (as_of - event_date).days)
    return exp(-days / 90.0)
