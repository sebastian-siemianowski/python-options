"""
Story 4.1: Unified Signal Output Contract.

Single source of truth for signal structure used by both terminal
rendering (Rich tables) and API serialization (Pydantic).

Usage:
    from decision.signal_output import SignalOutput, HorizonForecast
    signal = SignalOutput(symbol="AAPL", sector="Technology", ...)
    json_str = signal.to_json()
    signal2 = SignalOutput.from_json(json_str)
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class HorizonForecast:
    """Forecast for a single horizon."""
    horizon_days: int
    point_forecast_pct: float
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    direction_label: str = "neutral"
    confidence_score: float = 0.0
    model_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HorizonForecast":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SignalOutput:
    """
    Unified signal output contract.

    Both terminal (Rich) and API (Pydantic) render from this single structure.
    """
    symbol: str
    sector: str = ""
    crash_risk: float = 0.0
    momentum: float = 0.0
    horizon_forecasts: Dict[int, HorizonForecast] = field(default_factory=dict)
    confidence: float = 0.0
    regime: str = ""
    model_explanation: str = ""
    generated_at: str = ""
    data_version: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = asdict(self)
        return json.dumps(d, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SignalOutput":
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        hf = {}
        for k, v in d.get("horizon_forecasts", {}).items():
            hf[int(k)] = HorizonForecast.from_dict(v)
        d["horizon_forecasts"] = hf
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert to plain dict (for API serialization)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalOutput":
        """Build from dict."""
        hf = {}
        for k, v in d.get("horizon_forecasts", {}).items():
            if isinstance(v, dict):
                hf[int(k)] = HorizonForecast.from_dict(v)
            else:
                hf[int(k)] = v
        d = dict(d)
        d["horizon_forecasts"] = hf
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_rich_row(self) -> list:
        """
        Return list of values for Rich table rendering.
        Format: [symbol, sector, regime, confidence, crash_risk, 7d, 30d, 90d]
        """
        def _fmt(h: int) -> str:
            hf = self.horizon_forecasts.get(h)
            if hf is None:
                return "--"
            return f"{hf.point_forecast_pct:+.2f}%"

        return [
            self.symbol,
            self.sector,
            self.regime,
            f"{self.confidence:.0%}",
            f"{self.crash_risk:.1%}",
            _fmt(7),
            _fmt(30),
            _fmt(90),
        ]

    def get_direction_label(self, horizon: int = 7) -> str:
        """Get direction label for a horizon."""
        hf = self.horizon_forecasts.get(horizon)
        if hf is None:
            return "neutral"
        return hf.direction_label
