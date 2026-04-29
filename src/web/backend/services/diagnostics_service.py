"""
Diagnostics service — PIT diagnostics, model metrics, calibration status.

Provides structured data for the web diagnostics panel, equivalent to `make diag` / `make diag-pit`.
"""

import json
import os
import sys
import glob
from math import isfinite
from typing import Any, Dict, List, Optional
from datetime import datetime

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
TUNE_DIR = os.path.join(SRC_DIR, "data", "tune")
CACHE_DIR = os.path.join(SRC_DIR, "data", "cache")
DIAG_CACHE = os.path.join(CACHE_DIR, "diagnostics.json")
CALIB_FAILURES = os.path.join(SRC_DIR, "data", "calibration", "calibration_failures.json")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _global_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current global tune payload, tolerating legacy flat caches."""
    g = data.get("global")
    return g if isinstance(g, dict) else data


def _model_weights(g: Dict[str, Any]) -> Dict[str, float]:
    """Return BMA weights from either the legacy or current cache key."""
    weights = g.get("model_weights")
    if not isinstance(weights, dict) or not weights:
        weights = g.get("model_posterior")
    return weights if isinstance(weights, dict) else {}


def _models_block(data: Dict[str, Any], g: Dict[str, Any]) -> Dict[str, Any]:
    """Return fitted model diagnostics from the current or legacy cache shape."""
    models = g.get("models")
    if not isinstance(models, dict) or not models:
        models = data.get("models")
    return models if isinstance(models, dict) else {}


def _metric_number(value: Any, digits: Optional[int] = None) -> Optional[float]:
    """Return a JSON-safe finite float, preserving missing metrics as null."""
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not isfinite(result):
        return None
    return round(result, digits) if digits is not None else result


def _mean(values: List[float], digits: Optional[int] = None) -> Optional[float]:
    if not values:
        return None
    result = sum(values) / len(values)
    return round(result, digits) if digits is not None else result


def _model_family(model_name: str) -> str:
    """Classify model names dynamically for frontend grouping."""
    if model_name.startswith("phi_student_t_unified_improved"):
        return "Unified Student-t Improved"
    if model_name.startswith("phi_student_t_unified"):
        return "Unified Student-t"
    if model_name.startswith("phi_student_t_improved"):
        return "Student-t Improved"
    if model_name.startswith("phi_student_t"):
        return "Student-t"
    if model_name.startswith("rv_q_phi_gaussian"):
        return "RV-Q Gaussian"
    if model_name.startswith("rv_q_student_t"):
        return "RV-Q Student-t"
    if "gaussian_unified" in model_name:
        return "Unified Gaussian"
    if "gaussian" in model_name:
        return "Gaussian"
    return "Other"


def get_pit_summary() -> Dict[str, Any]:
    """
    Get PIT calibration summary for all tuned assets.
    
    Returns per-asset PIT status with model details, similar to `make diag-pit`.
    """
    if not os.path.isdir(TUNE_DIR):
        return {"assets": [], "total": 0, "passing": 0, "failing": 0}

    assets = []
    for filepath in sorted(glob.glob(os.path.join(TUNE_DIR, "*.json"))):
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        g = _global_payload(data)
        meta = data.get("meta", {})
        models = _models_block(data, g)
        bma_weights = _model_weights(g)

        # Extract per-model metrics
        model_metrics = []
        for model_name, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            metrics = model_data.get("metrics", model_data)
            model_metrics.append({
                "model": model_name,
                "bic": metrics.get("bic"),
                "crps": metrics.get("crps"),
                "hyvarinen": metrics.get("hyvarinen_score") or metrics.get("hyvarinen"),
                "pit_ks_pvalue": metrics.get("pit_ks_pvalue"),
                "ad_pvalue": metrics.get("ad_pvalue"),
                "histogram_mad": metrics.get("histogram_mad"),
                "weight": bma_weights.get(model_name, metrics.get("weight", 0)),
                "nu": metrics.get("nu"),
                "phi": metrics.get("phi"),
            })

        # Sort by weight descending
        model_metrics.sort(key=lambda m: m.get("weight") or 0, reverse=True)

        best_model = g.get("best_model", "unknown")

        asset_entry = {
            "symbol": symbol,
            "best_model": best_model,
            "pit_grade": g.get("pit_calibration_grade", "N/A"),
            "ad_stat": g.get("ad_stat"),
            "ad_critical": g.get("ad_critical_5pct"),
            "ad_pass": g.get("ad_pass"),
            "pit_ks_pvalue": g.get("pit_ks_pvalue"),
            "num_models": len(bma_weights),
            "bma_weights": bma_weights,
            "models": model_metrics,
            "regime": g.get("regime"),
            "last_tuned": meta.get("timestamp", "unknown"),
        }
        assets.append(asset_entry)

    passing = sum(1 for a in assets if a.get("ad_pass") is True)
    failing = sum(1 for a in assets if a.get("ad_pass") is False)
    unknown = len(assets) - passing - failing

    return {
        "assets": assets,
        "total": len(assets),
        "passing": passing,
        "failing": failing,
        "unknown": unknown,
        "computed_at": datetime.now().isoformat(),
    }


def get_calibration_failures() -> Dict[str, Any]:
    """
    Get calibration failures from the calibration_failures.json file.
    
    This is the same data used by `make calibrate`.
    """
    if not os.path.isfile(CALIB_FAILURES):
        return {"failures": [], "count": 0, "file_exists": False}

    try:
        with open(CALIB_FAILURES, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"failures": [], "count": 0, "file_exists": True, "error": "corrupt file"}

    failures = []
    if isinstance(data, dict):
        for symbol, info in data.items():
            entry = {"symbol": symbol}
            if isinstance(info, dict):
                entry.update(info)
            failures.append(entry)
    elif isinstance(data, list):
        failures = data

    return {
        "failures": failures,
        "count": len(failures),
        "file_exists": True,
    }


def get_model_comparison() -> Dict[str, Any]:
    """
    Get model comparison data across all assets.
    
    Returns aggregated statistics per model type, showing win rates,
    average weights, calibration quality, and a compact asset/model matrix.
    """
    if not os.path.isdir(TUNE_DIR):
        return {"models": {}, "model_names": [], "families": [], "matrix_rows": [], "total_assets": 0}

    model_stats: Dict[str, Dict[str, Any]] = {}
    family_stats: Dict[str, Dict[str, Any]] = {}
    matrix_rows: List[Dict[str, Any]] = []
    total_assets = 0

    for filepath in sorted(glob.glob(os.path.join(TUNE_DIR, "*.json"))):
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        total_assets += 1
        g = _global_payload(data)
        best_model = g.get("best_model", "unknown")
        bma_weights = _model_weights(g)
        models_block = _models_block(data, g)

        row_models: Dict[str, Any] = {}
        model_names = set(bma_weights.keys()) | set(models_block.keys())

        for model_name in sorted(model_names):
            model_data = models_block.get(model_name)
            metrics = model_data.get("metrics", model_data) if isinstance(model_data, dict) else {}
            weight = _metric_number(bma_weights.get(model_name, metrics.get("weight", 0.0)), 8) or 0.0
            bic = _metric_number(metrics.get("bic"))
            crps = _metric_number(metrics.get("crps"))
            hyvarinen = _metric_number(metrics.get("hyvarinen_score") or metrics.get("hyvarinen"))
            pit_p = _metric_number(metrics.get("pit_ks_pvalue"))
            ad_p = _metric_number(metrics.get("ad_pvalue"))
            histogram_mad = _metric_number(metrics.get("histogram_mad"))
            family = _model_family(model_name)

            if model_name not in model_stats:
                model_stats[model_name] = {
                    "name": model_name,
                    "family": family,
                    "win_count": 0,
                    "total_weight": 0.0,
                    "appearances": 0,
                    "weights": [],
                    "bic_values": [],
                    "crps_values": [],
                    "hyvarinen_values": [],
                    "pit_values": [],
                    "ad_values": [],
                    "histogram_mad_values": [],
                    "symbol_weights": [],
                }
            stats = model_stats[model_name]
            stats["appearances"] += 1
            stats["total_weight"] += weight
            stats["weights"].append(weight)
            stats["symbol_weights"].append((symbol, weight))
            if bic is not None:
                stats["bic_values"].append(bic)
            if crps is not None:
                stats["crps_values"].append(crps)
            if hyvarinen is not None:
                stats["hyvarinen_values"].append(hyvarinen)
            if pit_p is not None:
                stats["pit_values"].append(pit_p)
            if ad_p is not None:
                stats["ad_values"].append(ad_p)
            if histogram_mad is not None:
                stats["histogram_mad_values"].append(histogram_mad)
            if model_name == best_model:
                stats["win_count"] += 1

            if family not in family_stats:
                family_stats[family] = {
                    "family": family,
                    "model_count": 0,
                    "appearances": 0,
                    "win_count": 0,
                    "total_weight": 0.0,
                    "models": set(),
                }
            fstats = family_stats[family]
            fstats["models"].add(model_name)
            fstats["appearances"] += 1
            fstats["win_count"] += 1 if model_name == best_model else 0
            fstats["total_weight"] += weight

            row_models[model_name] = {
                "family": family,
                "winner": model_name == best_model,
                "weight": weight,
                "bic": bic,
                "crps": crps,
                "hyvarinen": hyvarinen,
                "pit_ks_pvalue": pit_p,
                "ad_pvalue": ad_p,
                "histogram_mad": histogram_mad,
            }

        matrix_rows.append({
            "symbol": symbol,
            "best_model": best_model,
            "regime": g.get("regime"),
            "ad_pass": g.get("ad_pass"),
            "models": row_models,
        })

    for stats in model_stats.values():
        n = stats["appearances"]
        stats["avg_weight"] = round(stats["total_weight"] / n, 6) if n > 0 else 0
        stats["win_rate"] = round(stats["win_count"] / total_assets, 4) if total_assets > 0 else 0
        stats["appearance_rate"] = round(n / total_assets, 4) if total_assets > 0 else 0
        weights = stats.pop("weights")
        stats["max_weight"] = round(max(weights), 6) if weights else 0
        stats["min_weight"] = round(min(weights), 6) if weights else 0
        stats["avg_bic"] = _mean(stats.pop("bic_values"), 2)
        stats["avg_crps"] = _mean(stats.pop("crps_values"), 6)
        stats["avg_hyvarinen"] = _mean(stats.pop("hyvarinen_values"), 2)
        stats["avg_pit_p"] = _mean(stats.pop("pit_values"), 6)
        stats["avg_ad_p"] = _mean(stats.pop("ad_values"), 6)
        stats["avg_histogram_mad"] = _mean(stats.pop("histogram_mad_values"), 6)
        symbol_weights = stats.pop("symbol_weights")
        stats["top_symbols"] = [
            sym for sym, _ in sorted(symbol_weights, key=lambda item: item[1], reverse=True)[:8]
        ]

    sorted_model_items = sorted(
        model_stats.items(),
        key=lambda x: (-x[1]["win_count"], -x[1]["avg_weight"], x[0]),
    )
    sorted_models = dict(sorted_model_items)

    families = []
    for stats in family_stats.values():
        appearances = stats["appearances"]
        families.append({
            "family": stats["family"],
            "model_count": len(stats.pop("models")),
            "appearances": appearances,
            "win_count": stats["win_count"],
            "avg_weight": round(stats["total_weight"] / appearances, 6) if appearances else 0,
        })
    families.sort(key=lambda item: (-item["win_count"], -item["avg_weight"], item["family"]))

    return {
        "models": sorted_models,
        "model_names": [name for name, _ in sorted_model_items],
        "families": families,
        "matrix_rows": matrix_rows,
        "total_assets": total_assets,
        "computed_at": datetime.now().isoformat(),
    }


def get_regime_distribution() -> Dict[str, Any]:
    """Get the distribution of regime classifications across assets."""
    if not os.path.isdir(TUNE_DIR):
        return {"regimes": {}, "total": 0}

    regime_counts: Dict[str, int] = {}
    regime_assets: Dict[str, List[str]] = {}
    total = 0

    for filepath in sorted(glob.glob(os.path.join(TUNE_DIR, "*.json"))):
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        total += 1
        regime = data.get("global", {}).get("regime", "unknown")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        if regime not in regime_assets:
            regime_assets[regime] = []
        regime_assets[regime].append(symbol)

    regimes = {}
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        regimes[regime] = {
            "count": count,
            "percentage": round(count / total * 100, 1) if total > 0 else 0,
            "assets": regime_assets.get(regime, []),
        }

    return {
        "regimes": regimes,
        "total": total,
        "computed_at": datetime.now().isoformat(),
    }


def get_cross_asset_summary() -> Dict[str, Any]:
    """
    Build cross-asset summary matrices (assets × models) for PIT, CRPS, AD.

    Equivalent to the terminal `make diag` summary tables showing how each
    model scores across all assets.
    """
    if not os.path.isdir(TUNE_DIR):
        return {"rows": [], "models": [], "total": 0}

    # Collect all model names first
    all_models: set = set()
    asset_data: List[Dict[str, Any]] = []

    for filepath in sorted(glob.glob(os.path.join(TUNE_DIR, "*.json"))):
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        g = _global_payload(data)
        models_block = _models_block(data, g)
        bma_weights = _model_weights(g)

        model_scores: Dict[str, Dict[str, Any]] = {}
        model_names = set(models_block.keys()) | set(bma_weights.keys())
        for model_name in sorted(model_names):
            model_data = models_block.get(model_name)
            metrics = model_data.get("metrics", model_data) if isinstance(model_data, dict) else {}
            all_models.add(model_name)
            model_scores[model_name] = {
                "family": _model_family(model_name),
                "crps": _metric_number(metrics.get("crps")),
                "pit_ks_p": _metric_number(metrics.get("pit_ks_pvalue")),
                "ad_p": _metric_number(metrics.get("ad_pvalue")),
                "bic": _metric_number(metrics.get("bic")),
                "hyv": _metric_number(metrics.get("hyvarinen_score") or metrics.get("hyvarinen")),
                "weight": _metric_number(bma_weights.get(model_name, metrics.get("weight", 0.0)), 8) or 0.0,
            }

        asset_data.append({
            "symbol": symbol,
            "best_model": g.get("best_model", "unknown"),
            "regime": g.get("regime"),
            "ad_pass": g.get("ad_pass"),
            "model_scores": model_scores,
        })

    sorted_models = sorted(all_models)

    # Build summary rows
    rows = []
    for ad in asset_data:
        row: Dict[str, Any] = {
            "symbol": ad["symbol"],
            "best_model": ad["best_model"],
            "regime": ad["regime"],
            "ad_pass": ad["ad_pass"],
            "scores": {},
        }
        for m in sorted_models:
            row["scores"][m] = ad["model_scores"].get(m)
        rows.append(row)

    # Compute per-model averages
    model_averages: Dict[str, Dict[str, Any]] = {}
    for m in sorted_models:
        crps_vals = []
        pit_vals = []
        bic_vals = []
        weight_vals = []
        for ad in asset_data:
            sc = ad["model_scores"].get(m)
            if sc:
                if sc.get("crps") is not None:
                    crps_vals.append(sc["crps"])
                if sc.get("pit_ks_p") is not None:
                    pit_vals.append(sc["pit_ks_p"])
                if sc.get("bic") is not None:
                    bic_vals.append(sc["bic"])
                weight_vals.append(sc.get("weight", 0.0))
        model_averages[m] = {
            "family": _model_family(m),
            "avg_crps": round(sum(crps_vals) / len(crps_vals), 5) if crps_vals else None,
            "avg_pit_p": round(sum(pit_vals) / len(pit_vals), 4) if pit_vals else None,
            "avg_bic": round(sum(bic_vals) / len(bic_vals), 1) if bic_vals else None,
            "avg_weight": round(sum(weight_vals) / len(weight_vals), 6) if weight_vals else 0,
            "count": len(crps_vals),
        }

    return {
        "rows": rows,
        "models": sorted_models,
        "model_averages": model_averages,
        "total": len(rows),
        "computed_at": datetime.now().isoformat(),
    }
