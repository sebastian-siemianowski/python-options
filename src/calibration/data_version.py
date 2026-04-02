"""
Story 4.2: Data Version Tracking and Consistency Checksums.

Hash chain: prices -> tune -> signals.
Each stage includes the hash of its input so consistency can be verified.

Usage:
    from calibration.data_version import (
        compute_price_hash, compute_tune_hash, compute_signal_hash,
        verify_hash_chain
    )
"""
import hashlib
import json
import os
from typing import Optional


def compute_price_hash(prices_path: str, tail_lines: int = 20) -> str:
    """
    SHA256 of last N lines of prices CSV.
    Fast: reads only tail of file.
    """
    if not os.path.exists(prices_path):
        return ""
    
    with open(prices_path, "r") as f:
        lines = f.readlines()
    
    tail = lines[-tail_lines:] if len(lines) >= tail_lines else lines
    content = "".join(tail).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def compute_tune_hash(price_hash: str, tune_params: dict) -> str:
    """
    SHA256 of price_hash + serialized tune params.
    """
    payload = price_hash + json.dumps(tune_params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_signal_hash(tune_hash: str, signal_data: dict) -> str:
    """
    SHA256 of tune_hash + serialized signal output.
    """
    payload = tune_hash + json.dumps(signal_data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def verify_hash_chain(
    prices_path: str,
    tune_cache: dict,
    signal_cache: Optional[dict] = None,
) -> dict:
    """
    Verify the full hash chain: prices -> tune -> signals.
    
    Returns:
        {
            "prices_consistent": bool,
            "tune_consistent": bool,
            "signals_consistent": bool | None,
            "status": "green" | "amber" | "red",
            "details": str,
        }
    """
    result = {
        "prices_consistent": False,
        "tune_consistent": False,
        "signals_consistent": None,
        "status": "red",
        "details": "",
    }
    
    # Check prices -> tune link
    current_price_hash = compute_price_hash(prices_path)
    stored_price_hash = tune_cache.get("price_data_hash", "")
    
    if not current_price_hash:
        result["details"] = "Price file not found"
        return result
    
    result["prices_consistent"] = (current_price_hash == stored_price_hash)
    
    # Check tune hash
    stored_tune_hash = tune_cache.get("tune_hash", "")
    if stored_tune_hash:
        result["tune_consistent"] = True
    
    # Check tune -> signals link
    if signal_cache is not None:
        signal_tune_hash = signal_cache.get("tune_hash", "")
        result["signals_consistent"] = (signal_tune_hash == stored_tune_hash)
    
    # Determine overall status
    if result["prices_consistent"] and result["tune_consistent"]:
        if result["signals_consistent"] is None or result["signals_consistent"]:
            result["status"] = "green"
            result["details"] = "All hashes consistent"
        else:
            result["status"] = "amber"
            result["details"] = "Signals generated from stale tune"
    elif not result["prices_consistent"]:
        result["status"] = "red"
        result["details"] = "Price data updated since last tune"
    else:
        result["status"] = "amber"
        result["details"] = "Tune hash missing or inconsistent"
    
    return result
