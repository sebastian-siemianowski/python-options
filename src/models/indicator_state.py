"""Causal indicator state contracts for model-integrated variants.

Indicators in this module are not trading rules.  They are typed, causal state
inputs that a distributional model may choose to consume through a declared
channel: drift, variance, q, tail thickness, asymmetry, regime likelihood, or
calibration.  The registry here is intentionally metadata-first; indicator math
is added only after a feature has a declared model-use contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False


INDICATOR_CHANNELS: Tuple[str, ...] = (
    "mean",
    "variance",
    "tail",
    "asymmetry",
    "q",
    "regime",
    "calibration",
    "confidence",
)


@dataclass(frozen=True)
class IndicatorFeatureSpec:
    """Metadata contract for one causal model-state indicator feature."""

    name: str
    family: str
    required_columns: Tuple[str, ...]
    lookback: int
    lag: int
    channels: Tuple[str, ...]
    output_names: Tuple[str, ...]
    description: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("indicator feature name is required")
        if self.lookback < 1:
            raise ValueError(f"{self.name}: lookback must be positive")
        if self.lag < 1:
            raise ValueError(f"{self.name}: model inputs must be lagged by at least one bar")
        unknown = tuple(ch for ch in self.channels if ch not in INDICATOR_CHANNELS)
        if unknown:
            raise ValueError(f"{self.name}: unknown model-use channels {unknown}")
        if not self.output_names:
            raise ValueError(f"{self.name}: at least one output name is required")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(f"{self.name}: duplicate output names are not allowed")


@dataclass(frozen=True)
class IndicatorStateBundle:
    """Arrays plus metadata passed from feature generation into model fitting."""

    n_obs: int
    features: Mapping[str, np.ndarray]
    availability: Mapping[str, bool]
    spec_names: Tuple[str, ...]
    source_columns: Tuple[str, ...]

    def feature_matrix(self, output_names: Sequence[str]) -> np.ndarray:
        """Return a contiguous matrix for selected feature outputs."""
        cols = []
        for name in output_names:
            arr = self.features.get(name)
            if arr is None:
                raise KeyError(f"indicator feature output not present: {name}")
            cols.append(np.asarray(arr, dtype=np.float64))
        if not cols:
            return np.empty((self.n_obs, 0), dtype=np.float64)
        return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)


_FEATURE_SPECS: Dict[str, IndicatorFeatureSpec] = {
    "heikin_ashi_state": IndicatorFeatureSpec(
        name="heikin_ashi_state",
        family="heikin_ashi",
        required_columns=("open", "high", "low", "close"),
        lookback=2,
        lag=1,
        channels=("mean", "variance", "tail", "asymmetry", "q"),
        output_names=(
            "ha_body_ratio",
            "ha_color",
            "ha_upper_wick_ratio",
            "ha_lower_wick_ratio",
            "ha_run_length",
            "ha_close_dislocation",
        ),
        description="Lagged Heikin-Ashi candle state for drift, q, variance, and tail conditioning.",
    ),
    "atr_supertrend_state": IndicatorFeatureSpec(
        name="atr_supertrend_state",
        family="atr_supertrend",
        required_columns=("high", "low", "close"),
        lookback=14,
        lag=1,
        channels=("variance", "q", "regime", "confidence"),
        output_names=(
            "atr_z",
            "supertrend_side",
            "supertrend_flip",
            "supertrend_flip_age",
            "supertrend_band_distance",
        ),
        description="Lagged ATR/SuperTrend state for volatility and regime-likelihood conditioning.",
    ),
    "chandelier_exit_state": IndicatorFeatureSpec(
        name="chandelier_exit_state",
        family="chandelier",
        required_columns=("high", "low", "close"),
        lookback=22,
        lag=1,
        channels=("variance", "confidence", "regime"),
        output_names=(
            "chandelier_long_distance",
            "chandelier_short_distance",
            "chandelier_crowding",
            "chandelier_flip_age",
        ),
        description="Lagged Chandelier/ATR stop distance for variance and confidence conditioning.",
    ),
    "kama_efficiency_state": IndicatorFeatureSpec(
        name="kama_efficiency_state",
        family="kama",
        required_columns=("close",),
        lookback=30,
        lag=1,
        channels=("mean", "q", "regime"),
        output_names=("kama_efficiency", "kama_slope", "kama_distance"),
        description="Lagged KAMA efficiency and equilibrium distance for drift/q conditioning.",
    ),
    "adx_dmi_state": IndicatorFeatureSpec(
        name="adx_dmi_state",
        family="adx_dmi",
        required_columns=("high", "low", "close"),
        lookback=14,
        lag=1,
        channels=("q", "tail", "regime"),
        output_names=("adx_strength", "dmi_spread", "dmi_abs_spread"),
        description="Lagged ADX/DMI directional-strength state for q, tail, and regime conditioning.",
    ),
    "ichimoku_state": IndicatorFeatureSpec(
        name="ichimoku_state",
        family="ichimoku",
        required_columns=("high", "low", "close"),
        lookback=52,
        lag=26,
        channels=("mean", "regime", "q"),
        output_names=("ichimoku_cloud_distance", "ichimoku_cloud_thickness", "ichimoku_kijun_distance"),
        description="Properly lagged Ichimoku equilibrium/cloud state for model dynamics.",
    ),
    "donchian_breakout_state": IndicatorFeatureSpec(
        name="donchian_breakout_state",
        family="donchian",
        required_columns=("high", "low", "close"),
        lookback=55,
        lag=1,
        channels=("mean", "variance", "regime"),
        output_names=("donchian_position", "donchian_width_z", "donchian_breakout_age"),
        description="Lagged Donchian range and breakout state for drift/variance conditioning.",
    ),
    "bollinger_keltner_state": IndicatorFeatureSpec(
        name="bollinger_keltner_state",
        family="squeeze",
        required_columns=("high", "low", "close"),
        lookback=20,
        lag=1,
        channels=("variance", "tail", "q"),
        output_names=("bb_percentile", "bb_width_z", "keltner_squeeze", "squeeze_release_age"),
        description="Lagged Bollinger/Keltner compression state for variance-transition modeling.",
    ),
    "oscillator_exhaustion_state": IndicatorFeatureSpec(
        name="oscillator_exhaustion_state",
        family="oscillator",
        required_columns=("high", "low", "close"),
        lookback=14,
        lag=1,
        channels=("mean", "tail", "asymmetry", "calibration"),
        output_names=("rsi_z", "stoch_rsi_z", "williams_r_z", "oscillator_failure"),
        description="Lagged bounded oscillator exhaustion state for tail/asymmetry conditioning.",
    ),
    "macd_acceleration_state": IndicatorFeatureSpec(
        name="macd_acceleration_state",
        family="momentum",
        required_columns=("close",),
        lookback=35,
        lag=1,
        channels=("mean", "confidence"),
        output_names=("macd_z", "ppo_z", "trix_z", "momentum_acceleration"),
        description="Lagged multi-scale acceleration state for drift and confidence conditioning.",
    ),
    "volume_flow_state": IndicatorFeatureSpec(
        name="volume_flow_state",
        family="volume_flow",
        required_columns=("high", "low", "close", "volume"),
        lookback=21,
        lag=1,
        channels=("variance", "tail", "confidence"),
        output_names=("obv_z", "mfi_z", "cmf_z", "volume_z"),
        description="Lagged volume/flow state for variance, tail, and confidence conditioning.",
    ),
    "vwap_dislocation_state": IndicatorFeatureSpec(
        name="vwap_dislocation_state",
        family="vwap",
        required_columns=("high", "low", "close", "volume"),
        lookback=63,
        lag=1,
        channels=("mean", "variance", "confidence"),
        output_names=("vwap_distance", "vwap_band_z", "vwap_slope"),
        description="Lagged rolling VWAP equilibrium and dislocation state.",
    ),
    "persistence_state": IndicatorFeatureSpec(
        name="persistence_state",
        family="persistence",
        required_columns=("close",),
        lookback=126,
        lag=1,
        channels=("q", "regime", "tail"),
        output_names=("hurst_proxy", "fractal_dimension_proxy", "wavelet_energy_z"),
        description="Lagged persistence and impulse-energy state for q/regime/tail conditioning.",
    ),
    "relative_strength_state": IndicatorFeatureSpec(
        name="relative_strength_state",
        family="relative_strength",
        required_columns=("close",),
        lookback=63,
        lag=1,
        channels=("mean", "confidence", "regime"),
        output_names=("relative_strength_z", "breadth_context", "beta_adjusted_momentum"),
        description="Lagged cross-sectional context for drift, confidence, and regime conditioning.",
    ),
}

for _spec in _FEATURE_SPECS.values():
    _spec.validate()


def get_indicator_feature_spec(name: str) -> Optional[IndicatorFeatureSpec]:
    """Return one indicator feature spec by canonical name."""
    return _FEATURE_SPECS.get(name)


def get_indicator_feature_specs() -> Dict[str, IndicatorFeatureSpec]:
    """Return a copy of all registered indicator feature specs."""
    return dict(_FEATURE_SPECS)


def get_indicator_features_for_channel(channel: str) -> Tuple[IndicatorFeatureSpec, ...]:
    """Return specs that may feed the requested model channel."""
    if channel not in INDICATOR_CHANNELS:
        raise ValueError(f"unknown indicator model channel: {channel}")
    return tuple(spec for spec in _FEATURE_SPECS.values() if channel in spec.channels)


def output_names_for_specs(spec_names: Iterable[str]) -> Tuple[str, ...]:
    """Return flattened output names for registered indicator specs."""
    outputs = []
    for name in spec_names:
        spec = get_indicator_feature_spec(name)
        if spec is None:
            raise KeyError(f"unknown indicator feature spec: {name}")
        outputs.extend(spec.output_names)
    return tuple(outputs)


def channels_for_specs(spec_names: Iterable[str]) -> Tuple[str, ...]:
    """Return sorted model-use channels covered by the requested specs."""
    channels = set()
    for name in spec_names:
        spec = get_indicator_feature_spec(name)
        if spec is None:
            raise KeyError(f"unknown indicator feature spec: {name}")
        channels.update(spec.channels)
    return tuple(ch for ch in INDICATOR_CHANNELS if ch in channels)


def required_columns_for_specs(spec_names: Iterable[str]) -> Tuple[str, ...]:
    """Return sorted OHLCV columns required by the requested specs."""
    cols = set()
    for name in spec_names:
        spec = get_indicator_feature_spec(name)
        if spec is None:
            raise KeyError(f"unknown indicator feature spec: {name}")
        cols.update(spec.required_columns)
    order = ("open", "high", "low", "close", "volume")
    return tuple(col for col in order if col in cols)


def validate_source_columns(spec_names: Iterable[str], source_columns: Iterable[str]) -> None:
    """Raise if source OHLCV columns cannot support the requested specs."""
    available = {str(col).lower() for col in source_columns}
    missing = tuple(col for col in required_columns_for_specs(spec_names) if col not in available)
    if missing:
        raise ValueError(f"indicator source columns missing: {missing}")


def build_empty_indicator_state(
    n_obs: int,
    spec_names: Sequence[str],
    source_columns: Sequence[str] = (),
) -> IndicatorStateBundle:
    """Build a deterministic empty bundle for models that request metadata only."""
    if n_obs < 0:
        raise ValueError("n_obs must be non-negative")
    for name in spec_names:
        if name not in _FEATURE_SPECS:
            raise KeyError(f"unknown indicator feature spec: {name}")
    features = {
        output: np.full(n_obs, np.nan, dtype=np.float64)
        for output in output_names_for_specs(spec_names)
    }
    availability = {name: False for name in spec_names}
    return IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability=availability,
        spec_names=tuple(spec_names),
        source_columns=tuple(str(col).lower() for col in source_columns),
    )


def validate_indicator_state_bundle(bundle: IndicatorStateBundle) -> None:
    """Validate a bundle before it is passed into model fitting or sampling."""
    known_outputs = set(output_names_for_specs(bundle.spec_names))
    unknown = tuple(name for name in bundle.features if name not in known_outputs)
    if unknown:
        raise ValueError(f"indicator bundle contains unregistered outputs: {unknown}")
    for name in bundle.spec_names:
        if name not in _FEATURE_SPECS:
            raise KeyError(f"unknown indicator feature spec in bundle: {name}")
    for output_name, values in bundle.features.items():
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"{output_name}: indicator output must be 1D")
        if len(arr) != bundle.n_obs:
            raise ValueError(f"{output_name}: length {len(arr)} != n_obs {bundle.n_obs}")
        finite_or_nan = np.isfinite(arr) | np.isnan(arr)
        if not bool(np.all(finite_or_nan)):
            raise ValueError(f"{output_name}: contains non-finite non-NaN values")


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_ohlc_arrays(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> None:
    n = len(close)
    if len(open_) != n or len(high) != n or len(low) != n:
        raise ValueError("open/high/low/close arrays must have the same length")
    finite = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    if not bool(np.all(finite)):
        raise ValueError("OHLC arrays must be finite")
    if not bool(np.all(high >= low)):
        raise ValueError("high must be greater than or equal to low")


def _validate_hlcv_arrays(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> None:
    n = len(close)
    if len(high) != n or len(low) != n or len(volume) != n:
        raise ValueError("high/low/close/volume arrays must have the same length")
    finite = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & np.isfinite(volume)
    if not bool(np.all(finite)):
        raise ValueError("HLCV arrays must be finite")
    if not bool(np.all(high >= low)):
        raise ValueError("high must be greater than or equal to low")
    if not bool(np.all(volume >= 0.0)):
        raise ValueError("volume must be non-negative")


def _heikin_ashi_state_reference(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    body_ratio = np.full(n, np.nan, dtype=np.float64)
    color = np.full(n, np.nan, dtype=np.float64)
    upper_wick_ratio = np.full(n, np.nan, dtype=np.float64)
    lower_wick_ratio = np.full(n, np.nan, dtype=np.float64)
    run_length = np.full(n, np.nan, dtype=np.float64)
    close_dislocation = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return body_ratio, color, upper_wick_ratio, lower_wick_ratio, run_length, close_dislocation

    raw_body = np.zeros(n, dtype=np.float64)
    raw_color = np.zeros(n, dtype=np.float64)
    raw_upper = np.zeros(n, dtype=np.float64)
    raw_lower = np.zeros(n, dtype=np.float64)
    raw_run = np.zeros(n, dtype=np.float64)
    raw_disloc = np.zeros(n, dtype=np.float64)

    ha_close_prev = 0.25 * (open_[0] + high[0] + low[0] + close[0])
    ha_open_prev = 0.5 * (open_[0] + close[0])
    signed_run = 0.0
    prev_color = 0.0

    for i in range(n):
        ha_close = 0.25 * (open_[i] + high[i] + low[i] + close[i])
        if i == 0:
            ha_open = ha_open_prev
        else:
            ha_open = 0.5 * (ha_open_prev + ha_close_prev)
        ha_high = max(high[i], ha_open, ha_close)
        ha_low = min(low[i], ha_open, ha_close)
        ha_range = max(ha_high - ha_low, 1e-12)
        body = ha_close - ha_open
        current_color = 1.0 if body > 0.0 else -1.0 if body < 0.0 else 0.0
        if current_color == 0.0:
            signed_run = 0.0
        elif current_color == prev_color:
            signed_run += current_color
        else:
            signed_run = current_color

        raw_body[i] = body / ha_range
        raw_color[i] = current_color
        raw_upper[i] = (ha_high - max(ha_open, ha_close)) / ha_range
        raw_lower[i] = (min(ha_open, ha_close) - ha_low) / ha_range
        raw_run[i] = signed_run
        raw_disloc[i] = (close[i] - ha_close) / ha_range

        ha_open_prev = ha_open
        ha_close_prev = ha_close
        prev_color = current_color

    for i in range(lag, n):
        j = i - lag
        body_ratio[i] = raw_body[j]
        color[i] = raw_color[j]
        upper_wick_ratio[i] = raw_upper[j]
        lower_wick_ratio[i] = raw_lower[j]
        run_length[i] = raw_run[j]
        close_dislocation[i] = raw_disloc[j]
    return body_ratio, color, upper_wick_ratio, lower_wick_ratio, run_length, close_dislocation


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _heikin_ashi_state_kernel(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        body_ratio = np.empty(n, dtype=np.float64)
        color = np.empty(n, dtype=np.float64)
        upper_wick_ratio = np.empty(n, dtype=np.float64)
        lower_wick_ratio = np.empty(n, dtype=np.float64)
        run_length = np.empty(n, dtype=np.float64)
        close_dislocation = np.empty(n, dtype=np.float64)
        for i in range(n):
            body_ratio[i] = np.nan
            color[i] = np.nan
            upper_wick_ratio[i] = np.nan
            lower_wick_ratio[i] = np.nan
            run_length[i] = np.nan
            close_dislocation[i] = np.nan
        if n == 0:
            return body_ratio, color, upper_wick_ratio, lower_wick_ratio, run_length, close_dislocation

        raw_body = np.zeros(n, dtype=np.float64)
        raw_color = np.zeros(n, dtype=np.float64)
        raw_upper = np.zeros(n, dtype=np.float64)
        raw_lower = np.zeros(n, dtype=np.float64)
        raw_run = np.zeros(n, dtype=np.float64)
        raw_disloc = np.zeros(n, dtype=np.float64)

        ha_close_prev = 0.25 * (open_[0] + high[0] + low[0] + close[0])
        ha_open_prev = 0.5 * (open_[0] + close[0])
        signed_run = 0.0
        prev_color = 0.0

        for i in range(n):
            ha_close = 0.25 * (open_[i] + high[i] + low[i] + close[i])
            if i == 0:
                ha_open = ha_open_prev
            else:
                ha_open = 0.5 * (ha_open_prev + ha_close_prev)
            ha_high = high[i]
            if ha_open > ha_high:
                ha_high = ha_open
            if ha_close > ha_high:
                ha_high = ha_close
            ha_low = low[i]
            if ha_open < ha_low:
                ha_low = ha_open
            if ha_close < ha_low:
                ha_low = ha_close
            ha_range = ha_high - ha_low
            if ha_range < 1e-12:
                ha_range = 1e-12
            body = ha_close - ha_open
            current_color = 0.0
            if body > 0.0:
                current_color = 1.0
            elif body < 0.0:
                current_color = -1.0
            if current_color == 0.0:
                signed_run = 0.0
            elif current_color == prev_color:
                signed_run += current_color
            else:
                signed_run = current_color

            body_top = ha_open
            body_bottom = ha_close
            if ha_close > ha_open:
                body_top = ha_close
                body_bottom = ha_open

            raw_body[i] = body / ha_range
            raw_color[i] = current_color
            raw_upper[i] = (ha_high - body_top) / ha_range
            raw_lower[i] = (body_bottom - ha_low) / ha_range
            raw_run[i] = signed_run
            raw_disloc[i] = (close[i] - ha_close) / ha_range

            ha_open_prev = ha_open
            ha_close_prev = ha_close
            prev_color = current_color

        for i in range(lag, n):
            j = i - lag
            body_ratio[i] = raw_body[j]
            color[i] = raw_color[j]
            upper_wick_ratio[i] = raw_upper[j]
            lower_wick_ratio[i] = raw_lower[j]
            run_length[i] = raw_run[j]
            close_dislocation[i] = raw_disloc[j]
        return body_ratio, color, upper_wick_ratio, lower_wick_ratio, run_length, close_dislocation
else:
    _heikin_ashi_state_kernel = None


def compute_heikin_ashi_state(
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged Heikin-Ashi state arrays for model integration.

    The returned arrays are aligned with the input bars.  Values at ``t`` use
    Heikin-Ashi state from ``t-lag`` so the feature can be consumed by a model
    predicting the next observation without peeking at the target bar.
    """
    if lag < 1:
        raise ValueError("Heikin-Ashi model state must be lagged by at least one bar")
    open_arr = _as_float_array(open_, "open")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_ohlc_arrays(open_arr, high_arr, low_arr, close_arr)

    if use_numba and _heikin_ashi_state_kernel is not None:
        values = _heikin_ashi_state_kernel(open_arr, high_arr, low_arr, close_arr, int(lag))
    else:
        values = _heikin_ashi_state_reference(open_arr, high_arr, low_arr, close_arr, int(lag))

    outputs = get_indicator_feature_spec("heikin_ashi_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_heikin_ashi_drift_signal(
    heikin_ashi_state: Mapping[str, Sequence[float]],
    clip: float = 1.0,
) -> np.ndarray:
    """Collapse lagged Heikin-Ashi state into a bounded drift-input signal.

    The output is unitless and causal.  Model fitting multiplies this signal by
    contemporaneous volatility and a learned dimensionless weight, producing a
    state-equation input in returns units:

        ``u_t = w_ha * sigma_t * ha_drift_signal_t``.
    """
    body = _as_float_array(heikin_ashi_state.get("ha_body_ratio", ()), "ha_body_ratio")
    color = _as_float_array(heikin_ashi_state.get("ha_color", ()), "ha_color")
    upper = _as_float_array(heikin_ashi_state.get("ha_upper_wick_ratio", ()), "ha_upper_wick_ratio")
    lower = _as_float_array(heikin_ashi_state.get("ha_lower_wick_ratio", ()), "ha_lower_wick_ratio")
    run = _as_float_array(heikin_ashi_state.get("ha_run_length", ()), "ha_run_length")
    dislocation = _as_float_array(heikin_ashi_state.get("ha_close_dislocation", ()), "ha_close_dislocation")

    n = len(body)
    if not (len(color) == len(upper) == len(lower) == len(run) == len(dislocation) == n):
        raise ValueError("Heikin-Ashi state arrays must have the same length")

    body = np.nan_to_num(body, nan=0.0, posinf=0.0, neginf=0.0)
    color = np.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    run = np.nan_to_num(run, nan=0.0, posinf=0.0, neginf=0.0)
    dislocation = np.nan_to_num(dislocation, nan=0.0, posinf=0.0, neginf=0.0)

    persistence = np.tanh(run / 3.0)
    wick_rejection = np.clip(lower - upper, -1.0, 1.0)
    stretch_penalty = np.clip(dislocation, -1.5, 1.5)

    raw = (
        0.62 * np.clip(body, -1.0, 1.0)
        + 0.18 * np.clip(color, -1.0, 1.0)
        + 0.24 * persistence
        + 0.18 * wick_rejection
        - 0.14 * stretch_penalty
    )
    signal = np.tanh(1.35 * raw)
    if clip > 0:
        signal = np.clip(signal, -float(clip), float(clip))
    return np.ascontiguousarray(signal, dtype=np.float64)


def build_heikin_ashi_bundle(
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Heikin-Ashi indicator-state bundle."""
    features = compute_heikin_ashi_state(open_, high, low, close, lag=lag, use_numba=use_numba)
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"heikin_ashi_state": True},
        spec_names=("heikin_ashi_state",),
        source_columns=("open", "high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _validate_hlc_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> None:
    n = len(close)
    if len(high) != n or len(low) != n:
        raise ValueError("high/low/close arrays must have the same length")
    finite = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    if not bool(np.all(finite)):
        raise ValueError("HLC arrays must be finite")
    if not bool(np.all(high >= low)):
        raise ValueError("high must be greater than or equal to low")


def _atr_supertrend_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    atr_z = np.full(n, np.nan, dtype=np.float64)
    side = np.full(n, np.nan, dtype=np.float64)
    flip = np.full(n, np.nan, dtype=np.float64)
    flip_age = np.full(n, np.nan, dtype=np.float64)
    band_distance = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return atr_z, side, flip, flip_age, band_distance

    tr = np.empty(n, dtype=np.float64)
    atr = np.empty(n, dtype=np.float64)
    raw_z = np.zeros(n, dtype=np.float64)
    raw_side = np.zeros(n, dtype=np.float64)
    raw_flip = np.zeros(n, dtype=np.float64)
    raw_age = np.zeros(n, dtype=np.float64)
    raw_dist = np.zeros(n, dtype=np.float64)

    for i in range(n):
        hl = high[i] - low[i]
        if i == 0:
            tr[i] = hl
        else:
            tr[i] = max(hl, abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if i == 0:
            atr[i] = max(tr[i], 1e-12)
        else:
            atr[i] = ((period - 1.0) * atr[i - 1] + tr[i]) / period
            if atr[i] < 1e-12:
                atr[i] = 1e-12

        start = max(0, i - period + 1)
        log_slice = np.log(np.maximum(atr[start:i + 1], 1e-12))
        mean = float(np.mean(log_slice))
        std = float(np.std(log_slice))
        raw_z[i] = (math_log_safe(atr[i]) - mean) / std if std > 1e-12 else 0.0

    hl2 = 0.5 * (high + low)
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    trend_side = 1.0
    age = 0.0
    for i in range(n):
        if i == 0:
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            trend_side = 1.0 if close[i] >= hl2[i] else -1.0
            raw_flip[i] = 0.0
            age = 0.0
        else:
            final_upper[i] = upper[i] if (upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
            final_lower[i] = lower[i] if (lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
            prev_side = trend_side
            if prev_side < 0.0 and close[i] > final_upper[i]:
                trend_side = 1.0
            elif prev_side > 0.0 and close[i] < final_lower[i]:
                trend_side = -1.0
            raw_flip[i] = 1.0 if trend_side != prev_side else 0.0
            age = 0.0 if raw_flip[i] > 0.0 else age + 1.0

        raw_side[i] = trend_side
        raw_age[i] = min(age / max(float(period), 1.0), 6.0)
        denom = max(atr[i], 1e-12)
        if trend_side > 0.0:
            raw_dist[i] = np.clip((close[i] - final_lower[i]) / denom, -6.0, 6.0)
        else:
            raw_dist[i] = np.clip((final_upper[i] - close[i]) / denom, -6.0, 6.0)

    for i in range(lag, n):
        j = i - lag
        atr_z[i] = raw_z[j]
        side[i] = raw_side[j]
        flip[i] = raw_flip[j]
        flip_age[i] = raw_age[j]
        band_distance[i] = raw_dist[j]
    return atr_z, side, flip, flip_age, band_distance


def math_log_safe(value: float) -> float:
    return float(np.log(max(float(value), 1e-12)))


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _atr_supertrend_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        multiplier: float,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        atr_z = np.empty(n, dtype=np.float64)
        side = np.empty(n, dtype=np.float64)
        flip = np.empty(n, dtype=np.float64)
        flip_age = np.empty(n, dtype=np.float64)
        band_distance = np.empty(n, dtype=np.float64)
        for i in range(n):
            atr_z[i] = np.nan
            side[i] = np.nan
            flip[i] = np.nan
            flip_age[i] = np.nan
            band_distance[i] = np.nan
        if n == 0:
            return atr_z, side, flip, flip_age, band_distance

        tr = np.empty(n, dtype=np.float64)
        atr = np.empty(n, dtype=np.float64)
        raw_z = np.zeros(n, dtype=np.float64)
        raw_side = np.zeros(n, dtype=np.float64)
        raw_flip = np.zeros(n, dtype=np.float64)
        raw_age = np.zeros(n, dtype=np.float64)
        raw_dist = np.zeros(n, dtype=np.float64)
        hl2 = np.empty(n, dtype=np.float64)
        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        final_upper = np.empty(n, dtype=np.float64)
        final_lower = np.empty(n, dtype=np.float64)

        for i in range(n):
            hl = high[i] - low[i]
            if i == 0:
                tr[i] = hl
            else:
                v1 = abs(high[i] - close[i - 1])
                v2 = abs(low[i] - close[i - 1])
                tr_i = hl
                if v1 > tr_i:
                    tr_i = v1
                if v2 > tr_i:
                    tr_i = v2
                tr[i] = tr_i
            if i == 0:
                atr[i] = tr[i] if tr[i] > 1e-12 else 1e-12
            else:
                atr[i] = ((period - 1.0) * atr[i - 1] + tr[i]) / period
                if atr[i] < 1e-12:
                    atr[i] = 1e-12

            start = i - period + 1
            if start < 0:
                start = 0
            count = i - start + 1
            mean = 0.0
            for j in range(start, i + 1):
                mean += np.log(atr[j] if atr[j] > 1e-12 else 1e-12)
            mean /= count
            var = 0.0
            for j in range(start, i + 1):
                lv = np.log(atr[j] if atr[j] > 1e-12 else 1e-12)
                diff = lv - mean
                var += diff * diff
            std = np.sqrt(var / count)
            raw_z[i] = (np.log(atr[i] if atr[i] > 1e-12 else 1e-12) - mean) / std if std > 1e-12 else 0.0

            hl2[i] = 0.5 * (high[i] + low[i])
            upper[i] = hl2[i] + multiplier * atr[i]
            lower[i] = hl2[i] - multiplier * atr[i]

        trend_side = 1.0
        age = 0.0
        for i in range(n):
            if i == 0:
                final_upper[i] = upper[i]
                final_lower[i] = lower[i]
                trend_side = 1.0 if close[i] >= hl2[i] else -1.0
                raw_flip[i] = 0.0
                age = 0.0
            else:
                if upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                    final_upper[i] = upper[i]
                else:
                    final_upper[i] = final_upper[i - 1]
                if lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                    final_lower[i] = lower[i]
                else:
                    final_lower[i] = final_lower[i - 1]
                prev_side = trend_side
                if prev_side < 0.0 and close[i] > final_upper[i]:
                    trend_side = 1.0
                elif prev_side > 0.0 and close[i] < final_lower[i]:
                    trend_side = -1.0
                raw_flip[i] = 1.0 if trend_side != prev_side else 0.0
                age = 0.0 if raw_flip[i] > 0.0 else age + 1.0

            raw_side[i] = trend_side
            denom = atr[i] if atr[i] > 1e-12 else 1e-12
            raw_age[i] = age / period if period > 0 else age
            if raw_age[i] > 6.0:
                raw_age[i] = 6.0
            if trend_side > 0.0:
                dist = (close[i] - final_lower[i]) / denom
            else:
                dist = (final_upper[i] - close[i]) / denom
            if dist < -6.0:
                dist = -6.0
            elif dist > 6.0:
                dist = 6.0
            raw_dist[i] = dist

        for i in range(lag, n):
            j = i - lag
            atr_z[i] = raw_z[j]
            side[i] = raw_side[j]
            flip[i] = raw_flip[j]
            flip_age[i] = raw_age[j]
            band_distance[i] = raw_dist[j]
        return atr_z, side, flip, flip_age, band_distance
else:
    _atr_supertrend_state_kernel = None


def compute_atr_supertrend_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    multiplier: float = 3.0,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged ATR and SuperTrend state for model integration."""
    if period < 2:
        raise ValueError("ATR period must be at least 2")
    if multiplier <= 0:
        raise ValueError("SuperTrend multiplier must be positive")
    if lag < 1:
        raise ValueError("ATR/SuperTrend model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)

    if use_numba and _atr_supertrend_state_kernel is not None:
        values = _atr_supertrend_state_kernel(
            high_arr, low_arr, close_arr, int(period), float(multiplier), int(lag)
        )
    else:
        values = _atr_supertrend_state_reference(
            high_arr, low_arr, close_arr, int(period), float(multiplier), int(lag)
        )
    outputs = get_indicator_feature_spec("atr_supertrend_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_atr_supertrend_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    multiplier: float = 3.0,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated ATR/SuperTrend indicator-state bundle."""
    features = compute_atr_supertrend_state(
        high, low, close, period=period, multiplier=multiplier, lag=lag, use_numba=use_numba
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"atr_supertrend_state": True},
        spec_names=("atr_supertrend_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _chandelier_exit_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    long_distance = np.full(n, np.nan, dtype=np.float64)
    short_distance = np.full(n, np.nan, dtype=np.float64)
    crowding = np.full(n, np.nan, dtype=np.float64)
    flip_age = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return long_distance, short_distance, crowding, flip_age

    tr = np.empty(n, dtype=np.float64)
    atr = np.empty(n, dtype=np.float64)
    raw_long = np.zeros(n, dtype=np.float64)
    raw_short = np.zeros(n, dtype=np.float64)
    raw_crowding = np.zeros(n, dtype=np.float64)
    raw_age = np.zeros(n, dtype=np.float64)
    side = 1.0
    age = 0.0

    for i in range(n):
        hl = high[i] - low[i]
        if i == 0:
            tr[i] = hl
        else:
            tr[i] = max(hl, abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if i == 0:
            atr[i] = max(tr[i], 1e-12)
        else:
            atr[i] = ((period - 1.0) * atr[i - 1] + tr[i]) / period
            atr[i] = max(atr[i], 1e-12)

        start = max(0, i - period + 1)
        highest = float(np.max(high[start:i + 1]))
        lowest = float(np.min(low[start:i + 1]))
        long_stop = highest - multiplier * atr[i]
        short_stop = lowest + multiplier * atr[i]
        denom = max(atr[i], 1e-12)

        long_d = np.clip((close[i] - long_stop) / denom, -6.0, 6.0)
        short_d = np.clip((short_stop - close[i]) / denom, -6.0, 6.0)
        prev_side = side
        if side > 0.0 and close[i] < long_stop:
            side = -1.0
        elif side < 0.0 and close[i] > short_stop:
            side = 1.0
        elif i == 0:
            mid = 0.5 * (highest + lowest)
            side = 1.0 if close[i] >= mid else -1.0
        age = 0.0 if side != prev_side else age + 1.0 if i > 0 else 0.0

        active_distance = long_d if side > 0.0 else short_d
        raw_long[i] = long_d
        raw_short[i] = short_d
        raw_crowding[i] = np.tanh((1.0 - active_distance) / 2.5)
        raw_age[i] = min(age / max(float(period), 1.0), 6.0)

    for i in range(lag, n):
        j = i - lag
        long_distance[i] = raw_long[j]
        short_distance[i] = raw_short[j]
        crowding[i] = raw_crowding[j]
        flip_age[i] = raw_age[j]
    return long_distance, short_distance, crowding, flip_age


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _chandelier_exit_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        multiplier: float,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        long_distance = np.empty(n, dtype=np.float64)
        short_distance = np.empty(n, dtype=np.float64)
        crowding = np.empty(n, dtype=np.float64)
        flip_age = np.empty(n, dtype=np.float64)
        for i in range(n):
            long_distance[i] = np.nan
            short_distance[i] = np.nan
            crowding[i] = np.nan
            flip_age[i] = np.nan
        if n == 0:
            return long_distance, short_distance, crowding, flip_age

        tr = np.empty(n, dtype=np.float64)
        atr = np.empty(n, dtype=np.float64)
        raw_long = np.zeros(n, dtype=np.float64)
        raw_short = np.zeros(n, dtype=np.float64)
        raw_crowding = np.zeros(n, dtype=np.float64)
        raw_age = np.zeros(n, dtype=np.float64)
        side = 1.0
        age = 0.0

        for i in range(n):
            hl = high[i] - low[i]
            if i == 0:
                tr[i] = hl
            else:
                v1 = abs(high[i] - close[i - 1])
                v2 = abs(low[i] - close[i - 1])
                tr_i = hl
                if v1 > tr_i:
                    tr_i = v1
                if v2 > tr_i:
                    tr_i = v2
                tr[i] = tr_i
            if i == 0:
                atr[i] = tr[i] if tr[i] > 1e-12 else 1e-12
            else:
                atr[i] = ((period - 1.0) * atr[i - 1] + tr[i]) / period
                if atr[i] < 1e-12:
                    atr[i] = 1e-12

            start = i - period + 1
            if start < 0:
                start = 0
            highest = high[start]
            lowest = low[start]
            for j in range(start + 1, i + 1):
                if high[j] > highest:
                    highest = high[j]
                if low[j] < lowest:
                    lowest = low[j]

            long_stop = highest - multiplier * atr[i]
            short_stop = lowest + multiplier * atr[i]
            denom = atr[i] if atr[i] > 1e-12 else 1e-12
            long_d = (close[i] - long_stop) / denom
            short_d = (short_stop - close[i]) / denom
            if long_d < -6.0:
                long_d = -6.0
            elif long_d > 6.0:
                long_d = 6.0
            if short_d < -6.0:
                short_d = -6.0
            elif short_d > 6.0:
                short_d = 6.0

            prev_side = side
            if side > 0.0 and close[i] < long_stop:
                side = -1.0
            elif side < 0.0 and close[i] > short_stop:
                side = 1.0
            elif i == 0:
                mid = 0.5 * (highest + lowest)
                side = 1.0 if close[i] >= mid else -1.0
            if i == 0:
                age = 0.0
            elif side != prev_side:
                age = 0.0
            else:
                age += 1.0

            active_distance = long_d if side > 0.0 else short_d
            raw_long[i] = long_d
            raw_short[i] = short_d
            raw_crowding[i] = np.tanh((1.0 - active_distance) / 2.5)
            raw_age[i] = age / period if period > 0 else age
            if raw_age[i] > 6.0:
                raw_age[i] = 6.0

        for i in range(lag, n):
            j = i - lag
            long_distance[i] = raw_long[j]
            short_distance[i] = raw_short[j]
            crowding[i] = raw_crowding[j]
            flip_age[i] = raw_age[j]
        return long_distance, short_distance, crowding, flip_age
else:
    _chandelier_exit_state_kernel = None


def compute_chandelier_exit_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 22,
    multiplier: float = 3.0,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged Chandelier/ATR stop-distance state for model integration."""
    if period < 2:
        raise ValueError("Chandelier period must be at least 2")
    if multiplier <= 0:
        raise ValueError("Chandelier multiplier must be positive")
    if lag < 1:
        raise ValueError("Chandelier model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)

    if use_numba and _chandelier_exit_state_kernel is not None:
        values = _chandelier_exit_state_kernel(
            high_arr, low_arr, close_arr, int(period), float(multiplier), int(lag)
        )
    else:
        values = _chandelier_exit_state_reference(
            high_arr, low_arr, close_arr, int(period), float(multiplier), int(lag)
        )
    outputs = get_indicator_feature_spec("chandelier_exit_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_chandelier_exit_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 22,
    multiplier: float = 3.0,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Chandelier/ATR stop-distance indicator-state bundle."""
    features = compute_chandelier_exit_state(
        high, low, close, period=period, multiplier=multiplier, lag=lag, use_numba=use_numba
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"chandelier_exit_state": True},
        spec_names=("chandelier_exit_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _kama_efficiency_state_reference(
    close: np.ndarray,
    er_period: int,
    fast_period: int,
    slow_period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    efficiency = np.full(n, np.nan, dtype=np.float64)
    slope = np.full(n, np.nan, dtype=np.float64)
    distance = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return efficiency, slope, distance

    raw_eff = np.zeros(n, dtype=np.float64)
    raw_slope = np.zeros(n, dtype=np.float64)
    raw_distance = np.zeros(n, dtype=np.float64)
    kama = np.empty(n, dtype=np.float64)
    kama[0] = close[0]
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)

    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        denom = max(abs(close[i - 1]), 1e-12)
        returns[i] = (close[i] - close[i - 1]) / denom

    for i in range(n):
        start = max(0, i - er_period)
        travel = abs(close[i] - close[start])
        path = 0.0
        for j in range(start + 1, i + 1):
            path += abs(close[j] - close[j - 1])
        er = travel / path if path > 1e-12 else 0.0
        er = float(np.clip(er, 0.0, 1.0))
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        if i > 0:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

        vol_start = max(1, i - slow_period + 1)
        if i >= vol_start:
            scale = float(np.std(returns[vol_start:i + 1]))
        else:
            scale = 0.0
        scale = max(scale, 1e-6)
        prev_close = close[i - 1] if i > 0 else close[i]
        prev_kama = kama[i - 1] if i > 0 else kama[i]
        slope_ret = (kama[i] - prev_kama) / max(abs(prev_close), 1e-12)
        dist_ret = (close[i] - kama[i]) / max(abs(close[i]), 1e-12)
        raw_eff[i] = er
        raw_slope[i] = float(np.clip(slope_ret / scale, -6.0, 6.0))
        raw_distance[i] = float(np.clip(dist_ret / scale, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        efficiency[i] = raw_eff[j]
        slope[i] = raw_slope[j]
        distance[i] = raw_distance[j]
    return efficiency, slope, distance


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _kama_efficiency_state_kernel(
        close: np.ndarray,
        er_period: int,
        fast_period: int,
        slow_period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        efficiency = np.empty(n, dtype=np.float64)
        slope = np.empty(n, dtype=np.float64)
        distance = np.empty(n, dtype=np.float64)
        for i in range(n):
            efficiency[i] = np.nan
            slope[i] = np.nan
            distance[i] = np.nan
        if n == 0:
            return efficiency, slope, distance

        raw_eff = np.zeros(n, dtype=np.float64)
        raw_slope = np.zeros(n, dtype=np.float64)
        raw_distance = np.zeros(n, dtype=np.float64)
        kama = np.empty(n, dtype=np.float64)
        returns = np.zeros(n, dtype=np.float64)
        kama[0] = close[0]
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)

        for i in range(1, n):
            denom = abs(close[i - 1])
            if denom < 1e-12:
                denom = 1e-12
            returns[i] = (close[i] - close[i - 1]) / denom

        for i in range(n):
            start = i - er_period
            if start < 0:
                start = 0
            travel = abs(close[i] - close[start])
            path = 0.0
            for j in range(start + 1, i + 1):
                path += abs(close[j] - close[j - 1])
            er = travel / path if path > 1e-12 else 0.0
            if er < 0.0:
                er = 0.0
            elif er > 1.0:
                er = 1.0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            if i > 0:
                kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])

            vol_start = i - slow_period + 1
            if vol_start < 1:
                vol_start = 1
            count = i - vol_start + 1
            scale = 0.0
            if count > 0:
                mean = 0.0
                for j in range(vol_start, i + 1):
                    mean += returns[j]
                mean /= count
                var = 0.0
                for j in range(vol_start, i + 1):
                    diff = returns[j] - mean
                    var += diff * diff
                scale = np.sqrt(var / count)
            if scale < 1e-6:
                scale = 1e-6

            prev_close = close[i - 1] if i > 0 else close[i]
            prev_kama = kama[i - 1] if i > 0 else kama[i]
            denom_prev = abs(prev_close)
            if denom_prev < 1e-12:
                denom_prev = 1e-12
            denom_now = abs(close[i])
            if denom_now < 1e-12:
                denom_now = 1e-12
            slope_z = ((kama[i] - prev_kama) / denom_prev) / scale
            dist_z = ((close[i] - kama[i]) / denom_now) / scale
            if slope_z < -6.0:
                slope_z = -6.0
            elif slope_z > 6.0:
                slope_z = 6.0
            if dist_z < -6.0:
                dist_z = -6.0
            elif dist_z > 6.0:
                dist_z = 6.0
            raw_eff[i] = er
            raw_slope[i] = slope_z
            raw_distance[i] = dist_z

        for i in range(lag, n):
            j = i - lag
            efficiency[i] = raw_eff[j]
            slope[i] = raw_slope[j]
            distance[i] = raw_distance[j]
        return efficiency, slope, distance
else:
    _kama_efficiency_state_kernel = None


def compute_kama_efficiency_state(
    close: Sequence[float],
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged KAMA efficiency, slope, and equilibrium distance state."""
    if er_period < 2:
        raise ValueError("KAMA efficiency period must be at least 2")
    if fast_period < 1 or slow_period <= fast_period:
        raise ValueError("KAMA periods must satisfy 1 <= fast_period < slow_period")
    if lag < 1:
        raise ValueError("KAMA model state must be lagged by at least one bar")
    close_arr = _as_float_array(close, "close")
    if not bool(np.all(np.isfinite(close_arr))):
        raise ValueError("close array must be finite")

    if use_numba and _kama_efficiency_state_kernel is not None:
        values = _kama_efficiency_state_kernel(
            close_arr, int(er_period), int(fast_period), int(slow_period), int(lag)
        )
    else:
        values = _kama_efficiency_state_reference(
            close_arr, int(er_period), int(fast_period), int(slow_period), int(lag)
        )
    outputs = get_indicator_feature_spec("kama_efficiency_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_kama_equilibrium_signal(
    kama_state: Mapping[str, Sequence[float]],
    clip: float = 1.0,
) -> np.ndarray:
    """Collapse KAMA state into a bounded mean-reversion input signal."""
    efficiency = _as_float_array(kama_state.get("kama_efficiency", ()), "kama_efficiency")
    slope = _as_float_array(kama_state.get("kama_slope", ()), "kama_slope")
    distance = _as_float_array(kama_state.get("kama_distance", ()), "kama_distance")
    n = len(efficiency)
    if len(slope) != n or len(distance) != n:
        raise ValueError("KAMA state arrays must have the same length")

    efficiency = np.clip(np.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
    trend_gate = np.clip(efficiency, 0.0, 1.0)
    mr_gate = 1.0 - trend_gate
    raw = 0.45 * np.tanh(slope / 2.0) - 0.65 * mr_gate * np.tanh(distance / 2.5)
    signal = np.tanh(1.25 * raw)
    if clip > 0:
        signal = np.clip(signal, -float(clip), float(clip))
    return np.ascontiguousarray(signal, dtype=np.float64)


def build_kama_efficiency_bundle(
    close: Sequence[float],
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated KAMA indicator-state bundle."""
    features = compute_kama_efficiency_state(
        close,
        er_period=er_period,
        fast_period=fast_period,
        slow_period=slow_period,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"kama_efficiency_state": True},
        spec_names=("kama_efficiency_state",),
        source_columns=("close",),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _adx_dmi_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    adx_strength = np.full(n, np.nan, dtype=np.float64)
    dmi_spread = np.full(n, np.nan, dtype=np.float64)
    dmi_abs_spread = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return adx_strength, dmi_spread, dmi_abs_spread

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    plus_s = np.zeros(n, dtype=np.float64)
    minus_s = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    raw_spread = np.zeros(n, dtype=np.float64)
    raw_abs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            tr[i] = max(high[i] - low[i], 1e-12)
            atr[i] = tr[i]
            continue
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0.0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0.0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]), 1e-12)
        atr[i] = ((period - 1.0) * atr[i - 1] + tr[i]) / period
        plus_s[i] = ((period - 1.0) * plus_s[i - 1] + plus_dm[i]) / period
        minus_s[i] = ((period - 1.0) * minus_s[i - 1] + minus_dm[i]) / period
        plus_di = 100.0 * plus_s[i] / max(atr[i], 1e-12)
        minus_di = 100.0 * minus_s[i] / max(atr[i], 1e-12)
        denom = plus_di + minus_di
        dx[i] = 100.0 * abs(plus_di - minus_di) / denom if denom > 1e-12 else 0.0
        adx[i] = ((period - 1.0) * adx[i - 1] + dx[i]) / period
        spread = np.clip((plus_di - minus_di) / 100.0, -1.0, 1.0)
        raw_spread[i] = spread
        raw_abs[i] = abs(spread)

    raw_strength = np.clip(adx / 100.0, 0.0, 1.0)
    for i in range(lag, n):
        j = i - lag
        adx_strength[i] = raw_strength[j]
        dmi_spread[i] = raw_spread[j]
        dmi_abs_spread[i] = raw_abs[j]
    return adx_strength, dmi_spread, dmi_abs_spread


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _adx_dmi_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        adx_strength = np.empty(n, dtype=np.float64)
        dmi_spread = np.empty(n, dtype=np.float64)
        dmi_abs_spread = np.empty(n, dtype=np.float64)
        for i in range(n):
            adx_strength[i] = np.nan
            dmi_spread[i] = np.nan
            dmi_abs_spread[i] = np.nan
        if n == 0:
            return adx_strength, dmi_spread, dmi_abs_spread

        tr = np.zeros(n, dtype=np.float64)
        plus_s = np.zeros(n, dtype=np.float64)
        minus_s = np.zeros(n, dtype=np.float64)
        atr = np.zeros(n, dtype=np.float64)
        dx = np.zeros(n, dtype=np.float64)
        adx = np.zeros(n, dtype=np.float64)
        raw_spread = np.zeros(n, dtype=np.float64)
        raw_abs = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i == 0:
                tr0 = high[i] - low[i]
                tr[i] = tr0 if tr0 > 1e-12 else 1e-12
                atr[i] = tr[i]
                continue
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm = up_move if up_move > down_move and up_move > 0.0 else 0.0
            minus_dm = down_move if down_move > up_move and down_move > 0.0 else 0.0
            tr_i = high[i] - low[i]
            v1 = abs(high[i] - close[i - 1])
            v2 = abs(low[i] - close[i - 1])
            if v1 > tr_i:
                tr_i = v1
            if v2 > tr_i:
                tr_i = v2
            if tr_i < 1e-12:
                tr_i = 1e-12
            tr[i] = tr_i
            atr[i] = ((period - 1.0) * atr[i - 1] + tr_i) / period
            plus_s[i] = ((period - 1.0) * plus_s[i - 1] + plus_dm) / period
            minus_s[i] = ((period - 1.0) * minus_s[i - 1] + minus_dm) / period
            denom_atr = atr[i] if atr[i] > 1e-12 else 1e-12
            plus_di = 100.0 * plus_s[i] / denom_atr
            minus_di = 100.0 * minus_s[i] / denom_atr
            denom = plus_di + minus_di
            dx_i = 100.0 * abs(plus_di - minus_di) / denom if denom > 1e-12 else 0.0
            dx[i] = dx_i
            adx[i] = ((period - 1.0) * adx[i - 1] + dx_i) / period
            spread = (plus_di - minus_di) / 100.0
            if spread < -1.0:
                spread = -1.0
            elif spread > 1.0:
                spread = 1.0
            raw_spread[i] = spread
            raw_abs[i] = abs(spread)

        for i in range(lag, n):
            j = i - lag
            strength = adx[j] / 100.0
            if strength < 0.0:
                strength = 0.0
            elif strength > 1.0:
                strength = 1.0
            adx_strength[i] = strength
            dmi_spread[i] = raw_spread[j]
            dmi_abs_spread[i] = raw_abs[j]
        return adx_strength, dmi_spread, dmi_abs_spread
else:
    _adx_dmi_state_kernel = None


def compute_adx_dmi_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged ADX/DMI state for model integration."""
    if period < 2:
        raise ValueError("ADX period must be at least 2")
    if lag < 1:
        raise ValueError("ADX/DMI model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)

    if use_numba and _adx_dmi_state_kernel is not None:
        values = _adx_dmi_state_kernel(high_arr, low_arr, close_arr, int(period), int(lag))
    else:
        values = _adx_dmi_state_reference(high_arr, low_arr, close_arr, int(period), int(lag))
    outputs = get_indicator_feature_spec("adx_dmi_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_adx_q_tail_conditioner(
    adx_state: Mapping[str, Sequence[float]],
    q_clip: Tuple[float, float] = (0.75, 1.35),
    tail_clip: Tuple[float, float] = (0.90, 1.20),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bounded q and tail-risk multipliers from ADX/DMI state.

    Strong, directional trends reduce latent drift churn slightly but keep a
    modest tail-risk premium for gap risk.  The primitive is intentionally
    conservative and is not applied unless a model variant explicitly earns it.
    """
    strength = _as_float_array(adx_state.get("adx_strength", ()), "adx_strength")
    abs_spread = _as_float_array(adx_state.get("dmi_abs_spread", ()), "dmi_abs_spread")
    if len(abs_spread) != len(strength):
        raise ValueError("ADX state arrays must have the same length")
    strength = np.clip(np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    abs_spread = np.clip(np.nan_to_num(abs_spread, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    trend_quality = strength * np.sqrt(abs_spread)
    chop = 1.0 - trend_quality
    q_mult = 1.0 + 0.28 * chop - 0.18 * trend_quality
    tail_mult = 1.0 + 0.16 * strength * (0.35 + abs_spread)
    q_mult = np.clip(q_mult, float(q_clip[0]), float(q_clip[1]))
    tail_mult = np.clip(tail_mult, float(tail_clip[0]), float(tail_clip[1]))
    return (
        np.ascontiguousarray(q_mult, dtype=np.float64),
        np.ascontiguousarray(tail_mult, dtype=np.float64),
    )


def build_adx_dmi_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated ADX/DMI indicator-state bundle."""
    features = compute_adx_dmi_state(high, low, close, period=period, lag=lag, use_numba=use_numba)
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"adx_dmi_state": True},
        spec_names=("adx_dmi_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _rolling_midpoint(high: np.ndarray, low: np.ndarray, end_idx: int, window: int) -> float:
    start = max(0, end_idx - window + 1)
    return 0.5 * (float(np.max(high[start:end_idx + 1])) + float(np.min(low[start:end_idx + 1])))


def _rolling_return_scale(close: np.ndarray, end_idx: int, window: int) -> float:
    start = max(1, end_idx - window + 1)
    vals = []
    for j in range(start, end_idx + 1):
        denom = max(abs(close[j - 1]), 1e-12)
        vals.append((close[j] - close[j - 1]) / denom)
    if not vals:
        return 1e-6
    return max(float(np.std(np.asarray(vals, dtype=np.float64))), 1e-6)


def _ichimoku_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int,
    kijun_period: int,
    span_b_period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    cloud_distance = np.full(n, np.nan, dtype=np.float64)
    cloud_thickness = np.full(n, np.nan, dtype=np.float64)
    kijun_distance = np.full(n, np.nan, dtype=np.float64)
    raw_cloud_distance = np.zeros(n, dtype=np.float64)
    raw_cloud_thickness = np.zeros(n, dtype=np.float64)
    raw_kijun_distance = np.zeros(n, dtype=np.float64)

    for i in range(n):
        tenkan = _rolling_midpoint(high, low, i, tenkan_period)
        kijun = _rolling_midpoint(high, low, i, kijun_period)
        span_a = 0.5 * (tenkan + kijun)
        span_b = _rolling_midpoint(high, low, i, span_b_period)
        cloud_mid = 0.5 * (span_a + span_b)
        scale = _rolling_return_scale(close, i, kijun_period)
        price = max(abs(close[i]), 1e-12)
        raw_cloud_distance[i] = float(np.clip(((close[i] - cloud_mid) / price) / scale, -6.0, 6.0))
        raw_cloud_thickness[i] = float(np.clip((abs(span_a - span_b) / price) / scale, 0.0, 6.0))
        raw_kijun_distance[i] = float(np.clip(((close[i] - kijun) / price) / scale, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        cloud_distance[i] = raw_cloud_distance[j]
        cloud_thickness[i] = raw_cloud_thickness[j]
        kijun_distance[i] = raw_kijun_distance[j]
    return cloud_distance, cloud_thickness, kijun_distance


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _ichimoku_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan_period: int,
        kijun_period: int,
        span_b_period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        cloud_distance = np.empty(n, dtype=np.float64)
        cloud_thickness = np.empty(n, dtype=np.float64)
        kijun_distance = np.empty(n, dtype=np.float64)
        raw_cloud_distance = np.zeros(n, dtype=np.float64)
        raw_cloud_thickness = np.zeros(n, dtype=np.float64)
        raw_kijun_distance = np.zeros(n, dtype=np.float64)
        for i in range(n):
            cloud_distance[i] = np.nan
            cloud_thickness[i] = np.nan
            kijun_distance[i] = np.nan

        for i in range(n):
            start_t = i - tenkan_period + 1
            if start_t < 0:
                start_t = 0
            high_t = high[start_t]
            low_t = low[start_t]
            for j in range(start_t + 1, i + 1):
                if high[j] > high_t:
                    high_t = high[j]
                if low[j] < low_t:
                    low_t = low[j]
            tenkan = 0.5 * (high_t + low_t)

            start_k = i - kijun_period + 1
            if start_k < 0:
                start_k = 0
            high_k = high[start_k]
            low_k = low[start_k]
            for j in range(start_k + 1, i + 1):
                if high[j] > high_k:
                    high_k = high[j]
                if low[j] < low_k:
                    low_k = low[j]
            kijun = 0.5 * (high_k + low_k)

            start_b = i - span_b_period + 1
            if start_b < 0:
                start_b = 0
            high_b = high[start_b]
            low_b = low[start_b]
            for j in range(start_b + 1, i + 1):
                if high[j] > high_b:
                    high_b = high[j]
                if low[j] < low_b:
                    low_b = low[j]
            span_b = 0.5 * (high_b + low_b)
            span_a = 0.5 * (tenkan + kijun)
            cloud_mid = 0.5 * (span_a + span_b)

            scale_start = i - kijun_period + 1
            if scale_start < 1:
                scale_start = 1
            count = i - scale_start + 1
            scale = 1e-6
            if count > 0:
                mean = 0.0
                for j in range(scale_start, i + 1):
                    denom_prev = abs(close[j - 1])
                    if denom_prev < 1e-12:
                        denom_prev = 1e-12
                    mean += (close[j] - close[j - 1]) / denom_prev
                mean /= count
                var = 0.0
                for j in range(scale_start, i + 1):
                    denom_prev = abs(close[j - 1])
                    if denom_prev < 1e-12:
                        denom_prev = 1e-12
                    ret = (close[j] - close[j - 1]) / denom_prev
                    diff = ret - mean
                    var += diff * diff
                scale = np.sqrt(var / count)
                if scale < 1e-6:
                    scale = 1e-6
            price = abs(close[i])
            if price < 1e-12:
                price = 1e-12
            cd = ((close[i] - cloud_mid) / price) / scale
            ct = (abs(span_a - span_b) / price) / scale
            kd = ((close[i] - kijun) / price) / scale
            if cd < -6.0:
                cd = -6.0
            elif cd > 6.0:
                cd = 6.0
            if ct < 0.0:
                ct = 0.0
            elif ct > 6.0:
                ct = 6.0
            if kd < -6.0:
                kd = -6.0
            elif kd > 6.0:
                kd = 6.0
            raw_cloud_distance[i] = cd
            raw_cloud_thickness[i] = ct
            raw_kijun_distance[i] = kd

        for i in range(lag, n):
            j = i - lag
            cloud_distance[i] = raw_cloud_distance[j]
            cloud_thickness[i] = raw_cloud_thickness[j]
            kijun_distance[i] = raw_kijun_distance[j]
        return cloud_distance, cloud_thickness, kijun_distance
else:
    _ichimoku_state_kernel = None


def compute_ichimoku_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    lag: int = 26,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute properly lagged Ichimoku equilibrium/cloud model state."""
    if not (2 <= tenkan_period < kijun_period <= span_b_period):
        raise ValueError("Ichimoku periods must satisfy 2 <= tenkan < kijun <= span_b")
    if lag < 1:
        raise ValueError("Ichimoku model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)
    if use_numba and _ichimoku_state_kernel is not None:
        values = _ichimoku_state_kernel(
            high_arr, low_arr, close_arr,
            int(tenkan_period), int(kijun_period), int(span_b_period), int(lag),
        )
    else:
        values = _ichimoku_state_reference(
            high_arr, low_arr, close_arr,
            int(tenkan_period), int(kijun_period), int(span_b_period), int(lag),
        )
    outputs = get_indicator_feature_spec("ichimoku_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_ichimoku_equilibrium_signal(
    ichimoku_state: Mapping[str, Sequence[float]],
    clip: float = 1.0,
) -> np.ndarray:
    """Collapse lagged Ichimoku state into a bounded equilibrium input."""
    cloud_distance = _as_float_array(ichimoku_state.get("ichimoku_cloud_distance", ()), "ichimoku_cloud_distance")
    cloud_thickness = _as_float_array(ichimoku_state.get("ichimoku_cloud_thickness", ()), "ichimoku_cloud_thickness")
    kijun_distance = _as_float_array(ichimoku_state.get("ichimoku_kijun_distance", ()), "ichimoku_kijun_distance")
    n = len(cloud_distance)
    if len(cloud_thickness) != n or len(kijun_distance) != n:
        raise ValueError("Ichimoku state arrays must have the same length")
    cloud_distance = np.nan_to_num(cloud_distance, nan=0.0, posinf=0.0, neginf=0.0)
    cloud_thickness = np.nan_to_num(cloud_thickness, nan=0.0, posinf=0.0, neginf=0.0)
    kijun_distance = np.nan_to_num(kijun_distance, nan=0.0, posinf=0.0, neginf=0.0)
    thickness_gate = 1.0 / (1.0 + np.maximum(cloud_thickness, 0.0))
    raw = -0.42 * np.tanh(kijun_distance / 2.5) * thickness_gate - 0.28 * np.tanh(cloud_distance / 3.0)
    signal = np.tanh(raw)
    if clip > 0:
        signal = np.clip(signal, -float(clip), float(clip))
    return np.ascontiguousarray(signal, dtype=np.float64)


def build_ichimoku_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    lag: int = 26,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Ichimoku indicator-state bundle."""
    features = compute_ichimoku_state(
        high,
        low,
        close,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        span_b_period=span_b_period,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"ichimoku_state": True},
        spec_names=("ichimoku_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _donchian_breakout_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    position = np.full(n, np.nan, dtype=np.float64)
    width_z = np.full(n, np.nan, dtype=np.float64)
    breakout_age = np.full(n, np.nan, dtype=np.float64)
    raw_position = np.zeros(n, dtype=np.float64)
    raw_width_z = np.zeros(n, dtype=np.float64)
    raw_age = np.zeros(n, dtype=np.float64)
    log_width = np.zeros(n, dtype=np.float64)
    signed_age = 0.0

    for i in range(n):
        start = max(0, i - period + 1)
        hi = float(np.max(high[start:i + 1]))
        lo = float(np.min(low[start:i + 1]))
        width = max(hi - lo, 1e-12)
        raw_position[i] = float(np.clip(2.0 * (close[i] - lo) / width - 1.0, -1.0, 1.0))
        log_width[i] = math_log_safe(width / max(abs(close[i]), 1e-12))

        if i > 0:
            pstart = max(0, i - period)
            prior_hi = float(np.max(high[pstart:i]))
            prior_lo = float(np.min(low[pstart:i]))
            if close[i] > prior_hi:
                signed_age = 1.0 if signed_age <= 0.0 else signed_age + 1.0
            elif close[i] < prior_lo:
                signed_age = -1.0 if signed_age >= 0.0 else signed_age - 1.0
            elif signed_age > 0.0:
                signed_age += 1.0
            elif signed_age < 0.0:
                signed_age -= 1.0
        raw_age[i] = float(np.clip(signed_age / max(float(period), 1.0), -6.0, 6.0))

        z_start = max(0, i - period + 1)
        sample = log_width[z_start:i + 1]
        mean = float(np.mean(sample))
        std = float(np.std(sample))
        raw_width_z[i] = (log_width[i] - mean) / std if std > 1e-12 else 0.0
        raw_width_z[i] = float(np.clip(raw_width_z[i], -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        position[i] = raw_position[j]
        width_z[i] = raw_width_z[j]
        breakout_age[i] = raw_age[j]
    return position, width_z, breakout_age


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _donchian_breakout_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        position = np.empty(n, dtype=np.float64)
        width_z = np.empty(n, dtype=np.float64)
        breakout_age = np.empty(n, dtype=np.float64)
        raw_position = np.zeros(n, dtype=np.float64)
        raw_width_z = np.zeros(n, dtype=np.float64)
        raw_age = np.zeros(n, dtype=np.float64)
        log_width = np.zeros(n, dtype=np.float64)
        for i in range(n):
            position[i] = np.nan
            width_z[i] = np.nan
            breakout_age[i] = np.nan
        signed_age = 0.0
        for i in range(n):
            start = i - period + 1
            if start < 0:
                start = 0
            hi = high[start]
            lo = low[start]
            for j in range(start + 1, i + 1):
                if high[j] > hi:
                    hi = high[j]
                if low[j] < lo:
                    lo = low[j]
            width = hi - lo
            if width < 1e-12:
                width = 1e-12
            pos = 2.0 * (close[i] - lo) / width - 1.0
            if pos < -1.0:
                pos = -1.0
            elif pos > 1.0:
                pos = 1.0
            raw_position[i] = pos
            price = abs(close[i])
            if price < 1e-12:
                price = 1e-12
            log_width[i] = np.log(width / price if width / price > 1e-12 else 1e-12)

            if i > 0:
                pstart = i - period
                if pstart < 0:
                    pstart = 0
                prior_hi = high[pstart]
                prior_lo = low[pstart]
                for j in range(pstart + 1, i):
                    if high[j] > prior_hi:
                        prior_hi = high[j]
                    if low[j] < prior_lo:
                        prior_lo = low[j]
                if close[i] > prior_hi:
                    signed_age = 1.0 if signed_age <= 0.0 else signed_age + 1.0
                elif close[i] < prior_lo:
                    signed_age = -1.0 if signed_age >= 0.0 else signed_age - 1.0
                elif signed_age > 0.0:
                    signed_age += 1.0
                elif signed_age < 0.0:
                    signed_age -= 1.0
            age = signed_age / period if period > 0 else signed_age
            if age < -6.0:
                age = -6.0
            elif age > 6.0:
                age = 6.0
            raw_age[i] = age

            z_start = i - period + 1
            if z_start < 0:
                z_start = 0
            count = i - z_start + 1
            mean = 0.0
            for j in range(z_start, i + 1):
                mean += log_width[j]
            mean /= count
            var = 0.0
            for j in range(z_start, i + 1):
                diff = log_width[j] - mean
                var += diff * diff
            std = np.sqrt(var / count)
            z = (log_width[i] - mean) / std if std > 1e-12 else 0.0
            if z < -6.0:
                z = -6.0
            elif z > 6.0:
                z = 6.0
            raw_width_z[i] = z

        for i in range(lag, n):
            j = i - lag
            position[i] = raw_position[j]
            width_z[i] = raw_width_z[j]
            breakout_age[i] = raw_age[j]
        return position, width_z, breakout_age
else:
    _donchian_breakout_state_kernel = None


def compute_donchian_breakout_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 55,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged Donchian range and breakout state."""
    if period < 2:
        raise ValueError("Donchian period must be at least 2")
    if lag < 1:
        raise ValueError("Donchian model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)
    if use_numba and _donchian_breakout_state_kernel is not None:
        values = _donchian_breakout_state_kernel(high_arr, low_arr, close_arr, int(period), int(lag))
    else:
        values = _donchian_breakout_state_reference(high_arr, low_arr, close_arr, int(period), int(lag))
    outputs = get_indicator_feature_spec("donchian_breakout_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_turtle_breakout_quality(
    donchian_state: Mapping[str, Sequence[float]],
    clip: float = 1.0,
) -> np.ndarray:
    """Return a bounded breakout-quality model-context signal."""
    position = _as_float_array(donchian_state.get("donchian_position", ()), "donchian_position")
    width_z = _as_float_array(donchian_state.get("donchian_width_z", ()), "donchian_width_z")
    age = _as_float_array(donchian_state.get("donchian_breakout_age", ()), "donchian_breakout_age")
    n = len(position)
    if len(width_z) != n or len(age) != n:
        raise ValueError("Donchian state arrays must have the same length")
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    width_z = np.nan_to_num(width_z, nan=0.0, posinf=0.0, neginf=0.0)
    age = np.nan_to_num(age, nan=0.0, posinf=0.0, neginf=0.0)
    expansion = np.tanh(np.maximum(width_z, 0.0) / 2.0)
    follow_through = np.tanh(age)
    raw = 0.55 * position * expansion + 0.45 * follow_through
    signal = np.tanh(raw)
    if clip > 0:
        signal = np.clip(signal, -float(clip), float(clip))
    return np.ascontiguousarray(signal, dtype=np.float64)


def build_donchian_breakout_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 55,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Donchian breakout indicator-state bundle."""
    features = compute_donchian_breakout_state(high, low, close, period=period, lag=lag, use_numba=use_numba)
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"donchian_breakout_state": True},
        spec_names=("donchian_breakout_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _bollinger_keltner_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    bb_mult: float,
    keltner_mult: float,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    percentile = np.full(n, np.nan, dtype=np.float64)
    width_z = np.full(n, np.nan, dtype=np.float64)
    squeeze = np.full(n, np.nan, dtype=np.float64)
    release_age = np.full(n, np.nan, dtype=np.float64)
    raw_percentile = np.zeros(n, dtype=np.float64)
    raw_width_z = np.zeros(n, dtype=np.float64)
    raw_squeeze = np.zeros(n, dtype=np.float64)
    raw_release = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    log_width = np.zeros(n, dtype=np.float64)
    release = 0.0
    was_squeezed = False

    for i in range(n):
        hl = high[i] - low[i]
        if i == 0:
            tr = hl
            atr[i] = max(tr, 1e-12)
        else:
            tr = max(hl, abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            atr[i] = ((period - 1.0) * atr[i - 1] + tr) / period
            atr[i] = max(atr[i], 1e-12)
        start = max(0, i - period + 1)
        sample = close[start:i + 1]
        mean = float(np.mean(sample))
        std = float(np.std(sample))
        bb_half = max(bb_mult * std, 1e-12)
        bb_width = 2.0 * bb_half
        kel_width = 2.0 * keltner_mult * atr[i]
        lower = mean - bb_half
        upper = mean + bb_half
        raw_percentile[i] = float(np.clip((close[i] - lower) / max(upper - lower, 1e-12), 0.0, 1.0))
        log_width[i] = math_log_safe(bb_width / max(abs(close[i]), 1e-12))
        is_squeeze = bool(bb_width < kel_width)
        raw_squeeze[i] = 1.0 if is_squeeze else 0.0
        if is_squeeze:
            release = 0.0
            was_squeezed = True
        elif was_squeezed:
            release = release + 1.0 if release > 0.0 else 1.0
        raw_release[i] = min(release / max(float(period), 1.0), 6.0)

        w_sample = log_width[start:i + 1]
        w_mean = float(np.mean(w_sample))
        w_std = float(np.std(w_sample))
        raw_width_z[i] = (log_width[i] - w_mean) / w_std if w_std > 1e-12 else 0.0
        raw_width_z[i] = float(np.clip(raw_width_z[i], -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        percentile[i] = raw_percentile[j]
        width_z[i] = raw_width_z[j]
        squeeze[i] = raw_squeeze[j]
        release_age[i] = raw_release[j]
    return percentile, width_z, squeeze, release_age


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _bollinger_keltner_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        bb_mult: float,
        keltner_mult: float,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        percentile = np.empty(n, dtype=np.float64)
        width_z = np.empty(n, dtype=np.float64)
        squeeze = np.empty(n, dtype=np.float64)
        release_age = np.empty(n, dtype=np.float64)
        raw_percentile = np.zeros(n, dtype=np.float64)
        raw_width_z = np.zeros(n, dtype=np.float64)
        raw_squeeze = np.zeros(n, dtype=np.float64)
        raw_release = np.zeros(n, dtype=np.float64)
        atr = np.zeros(n, dtype=np.float64)
        log_width = np.zeros(n, dtype=np.float64)
        for i in range(n):
            percentile[i] = np.nan
            width_z[i] = np.nan
            squeeze[i] = np.nan
            release_age[i] = np.nan
        release = 0.0
        was_squeezed = False
        for i in range(n):
            hl = high[i] - low[i]
            if i == 0:
                tr = hl
                atr[i] = tr if tr > 1e-12 else 1e-12
            else:
                v1 = abs(high[i] - close[i - 1])
                v2 = abs(low[i] - close[i - 1])
                tr = hl
                if v1 > tr:
                    tr = v1
                if v2 > tr:
                    tr = v2
                atr[i] = ((period - 1.0) * atr[i - 1] + tr) / period
                if atr[i] < 1e-12:
                    atr[i] = 1e-12
            start = i - period + 1
            if start < 0:
                start = 0
            count = i - start + 1
            mean = 0.0
            for j in range(start, i + 1):
                mean += close[j]
            mean /= count
            var = 0.0
            for j in range(start, i + 1):
                diff = close[j] - mean
                var += diff * diff
            std = np.sqrt(var / count)
            bb_half = bb_mult * std
            if bb_half < 1e-12:
                bb_half = 1e-12
            bb_width = 2.0 * bb_half
            kel_width = 2.0 * keltner_mult * atr[i]
            lower = mean - bb_half
            upper = mean + bb_half
            pct = (close[i] - lower) / (upper - lower if upper > lower else 1e-12)
            if pct < 0.0:
                pct = 0.0
            elif pct > 1.0:
                pct = 1.0
            raw_percentile[i] = pct
            price = abs(close[i])
            if price < 1e-12:
                price = 1e-12
            lw = bb_width / price
            if lw < 1e-12:
                lw = 1e-12
            log_width[i] = np.log(lw)
            is_squeeze = bb_width < kel_width
            raw_squeeze[i] = 1.0 if is_squeeze else 0.0
            if is_squeeze:
                release = 0.0
                was_squeezed = True
            elif was_squeezed:
                release = release + 1.0 if release > 0.0 else 1.0
            rel = release / period if period > 0 else release
            if rel > 6.0:
                rel = 6.0
            raw_release[i] = rel

            w_mean = 0.0
            for j in range(start, i + 1):
                w_mean += log_width[j]
            w_mean /= count
            w_var = 0.0
            for j in range(start, i + 1):
                diff = log_width[j] - w_mean
                w_var += diff * diff
            w_std = np.sqrt(w_var / count)
            z = (log_width[i] - w_mean) / w_std if w_std > 1e-12 else 0.0
            if z < -6.0:
                z = -6.0
            elif z > 6.0:
                z = 6.0
            raw_width_z[i] = z

        for i in range(lag, n):
            j = i - lag
            percentile[i] = raw_percentile[j]
            width_z[i] = raw_width_z[j]
            squeeze[i] = raw_squeeze[j]
            release_age[i] = raw_release[j]
        return percentile, width_z, squeeze, release_age
else:
    _bollinger_keltner_state_kernel = None


def compute_bollinger_keltner_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 20,
    bb_mult: float = 2.0,
    keltner_mult: float = 1.5,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged Bollinger/Keltner squeeze state."""
    if period < 2:
        raise ValueError("Bollinger/Keltner period must be at least 2")
    if bb_mult <= 0 or keltner_mult <= 0:
        raise ValueError("Bollinger and Keltner multipliers must be positive")
    if lag < 1:
        raise ValueError("Bollinger/Keltner model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)
    if use_numba and _bollinger_keltner_state_kernel is not None:
        values = _bollinger_keltner_state_kernel(
            high_arr, low_arr, close_arr, int(period), float(bb_mult), float(keltner_mult), int(lag)
        )
    else:
        values = _bollinger_keltner_state_reference(
            high_arr, low_arr, close_arr, int(period), float(bb_mult), float(keltner_mult), int(lag)
        )
    outputs = get_indicator_feature_spec("bollinger_keltner_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_bollinger_variance_conditioner(
    squeeze_state: Mapping[str, Sequence[float]],
    clip: Tuple[float, float] = (0.80, 1.45),
) -> np.ndarray:
    """Return a bounded variance multiplier from squeeze/release context."""
    width_z = _as_float_array(squeeze_state.get("bb_width_z", ()), "bb_width_z")
    squeeze = _as_float_array(squeeze_state.get("keltner_squeeze", ()), "keltner_squeeze")
    release_age = _as_float_array(squeeze_state.get("squeeze_release_age", ()), "squeeze_release_age")
    n = len(width_z)
    if len(squeeze) != n or len(release_age) != n:
        raise ValueError("Bollinger/Keltner state arrays must have the same length")
    width_z = np.nan_to_num(width_z, nan=0.0, posinf=0.0, neginf=0.0)
    squeeze = np.clip(np.nan_to_num(squeeze, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    release_age = np.nan_to_num(release_age, nan=0.0, posinf=0.0, neginf=0.0)
    compression_discount = -0.12 * squeeze * np.tanh(np.maximum(-width_z, 0.0))
    release_premium = 0.28 * np.tanh(np.maximum(release_age, 0.0)) * (1.0 + 0.25 * np.maximum(width_z, 0.0))
    mult = 1.0 + compression_discount + release_premium
    return np.ascontiguousarray(np.clip(mult, float(clip[0]), float(clip[1])), dtype=np.float64)


def build_bollinger_keltner_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 20,
    bb_mult: float = 2.0,
    keltner_mult: float = 1.5,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Bollinger/Keltner squeeze indicator-state bundle."""
    features = compute_bollinger_keltner_state(
        high, low, close,
        period=period,
        bb_mult=bb_mult,
        keltner_mult=keltner_mult,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"bollinger_keltner_state": True},
        spec_names=("bollinger_keltner_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _oscillator_exhaustion_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    rsi_z = np.full(n, np.nan, dtype=np.float64)
    stoch_rsi_z = np.full(n, np.nan, dtype=np.float64)
    williams_z = np.full(n, np.nan, dtype=np.float64)
    failure = np.full(n, np.nan, dtype=np.float64)
    raw_rsi = np.zeros(n, dtype=np.float64)
    raw_stoch = np.zeros(n, dtype=np.float64)
    raw_williams = np.zeros(n, dtype=np.float64)
    raw_failure = np.zeros(n, dtype=np.float64)
    avg_gain = 0.0
    avg_loss = 0.0
    rsi = np.full(n, 50.0, dtype=np.float64)

    for i in range(n):
        if i > 0:
            change = close[i] - close[i - 1]
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            avg_gain = ((period - 1.0) * avg_gain + gain) / period
            avg_loss = ((period - 1.0) * avg_loss + loss) / period
        rs = avg_gain / max(avg_loss, 1e-12)
        rsi[i] = 100.0 - 100.0 / (1.0 + rs) if avg_loss > 1e-12 else 100.0 if avg_gain > 1e-12 else 50.0
        raw_rsi[i] = float(np.clip((rsi[i] - 50.0) / 25.0, -3.0, 3.0))

        start = max(0, i - period + 1)
        rsi_min = float(np.min(rsi[start:i + 1]))
        rsi_max = float(np.max(rsi[start:i + 1]))
        rsi_range = rsi_max - rsi_min
        stoch = (rsi[i] - rsi_min) / rsi_range if rsi_range > 1e-12 else 0.5
        raw_stoch[i] = float(np.clip(2.0 * stoch - 1.0, -1.0, 1.0))

        hi = float(np.max(high[start:i + 1]))
        lo = float(np.min(low[start:i + 1]))
        pos = (close[i] - lo) / max(hi - lo, 1e-12)
        raw_williams[i] = float(np.clip(2.0 * pos - 1.0, -1.0, 1.0))
        disagreement = raw_williams[i] - np.clip(raw_rsi[i] / 3.0, -1.0, 1.0)
        raw_failure[i] = float(np.tanh(disagreement))

    for i in range(lag, n):
        j = i - lag
        rsi_z[i] = raw_rsi[j]
        stoch_rsi_z[i] = raw_stoch[j]
        williams_z[i] = raw_williams[j]
        failure[i] = raw_failure[j]
    return rsi_z, stoch_rsi_z, williams_z, failure


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _oscillator_exhaustion_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        rsi_z = np.empty(n, dtype=np.float64)
        stoch_rsi_z = np.empty(n, dtype=np.float64)
        williams_z = np.empty(n, dtype=np.float64)
        failure = np.empty(n, dtype=np.float64)
        raw_rsi = np.zeros(n, dtype=np.float64)
        raw_stoch = np.zeros(n, dtype=np.float64)
        raw_williams = np.zeros(n, dtype=np.float64)
        raw_failure = np.zeros(n, dtype=np.float64)
        rsi = np.empty(n, dtype=np.float64)
        for i in range(n):
            rsi_z[i] = np.nan
            stoch_rsi_z[i] = np.nan
            williams_z[i] = np.nan
            failure[i] = np.nan
            rsi[i] = 50.0
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(n):
            if i > 0:
                change = close[i] - close[i - 1]
                gain = change if change > 0.0 else 0.0
                loss = -change if change < 0.0 else 0.0
                avg_gain = ((period - 1.0) * avg_gain + gain) / period
                avg_loss = ((period - 1.0) * avg_loss + loss) / period
            if avg_loss > 1e-12:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
            elif avg_gain > 1e-12:
                rsi[i] = 100.0
            else:
                rsi[i] = 50.0
            rz = (rsi[i] - 50.0) / 25.0
            if rz < -3.0:
                rz = -3.0
            elif rz > 3.0:
                rz = 3.0
            raw_rsi[i] = rz

            start = i - period + 1
            if start < 0:
                start = 0
            rsi_min = rsi[start]
            rsi_max = rsi[start]
            hi = high[start]
            lo = low[start]
            for j in range(start + 1, i + 1):
                if rsi[j] < rsi_min:
                    rsi_min = rsi[j]
                if rsi[j] > rsi_max:
                    rsi_max = rsi[j]
                if high[j] > hi:
                    hi = high[j]
                if low[j] < lo:
                    lo = low[j]
            denom_rsi = rsi_max - rsi_min
            stoch = (rsi[i] - rsi_min) / denom_rsi if denom_rsi > 1e-12 else 0.5
            srz = 2.0 * stoch - 1.0
            if srz < -1.0:
                srz = -1.0
            elif srz > 1.0:
                srz = 1.0
            raw_stoch[i] = srz
            denom_price = hi - lo
            pos = (close[i] - lo) / denom_price if denom_price > 1e-12 else 0.5
            wz = 2.0 * pos - 1.0
            if wz < -1.0:
                wz = -1.0
            elif wz > 1.0:
                wz = 1.0
            raw_williams[i] = wz
            rsi_unit = rz / 3.0
            if rsi_unit < -1.0:
                rsi_unit = -1.0
            elif rsi_unit > 1.0:
                rsi_unit = 1.0
            raw_failure[i] = np.tanh(wz - rsi_unit)

        for i in range(lag, n):
            j = i - lag
            rsi_z[i] = raw_rsi[j]
            stoch_rsi_z[i] = raw_stoch[j]
            williams_z[i] = raw_williams[j]
            failure[i] = raw_failure[j]
        return rsi_z, stoch_rsi_z, williams_z, failure
else:
    _oscillator_exhaustion_state_kernel = None


def compute_oscillator_exhaustion_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged RSI/StochRSI/Williams exhaustion state."""
    if period < 2:
        raise ValueError("Oscillator period must be at least 2")
    if lag < 1:
        raise ValueError("Oscillator model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    _validate_hlc_arrays(high_arr, low_arr, close_arr)
    if use_numba and _oscillator_exhaustion_state_kernel is not None:
        values = _oscillator_exhaustion_state_kernel(high_arr, low_arr, close_arr, int(period), int(lag))
    else:
        values = _oscillator_exhaustion_state_reference(high_arr, low_arr, close_arr, int(period), int(lag))
    outputs = get_indicator_feature_spec("oscillator_exhaustion_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def compute_williams_failed_breakout_signal(
    oscillator_state: Mapping[str, Sequence[float]],
    donchian_state: Mapping[str, Sequence[float]],
    clip: float = 1.0,
) -> np.ndarray:
    """Return bounded range-regime failed-breakout context."""
    williams = _as_float_array(oscillator_state.get("williams_r_z", ()), "williams_r_z")
    failure = _as_float_array(oscillator_state.get("oscillator_failure", ()), "oscillator_failure")
    position = _as_float_array(donchian_state.get("donchian_position", ()), "donchian_position")
    age = _as_float_array(donchian_state.get("donchian_breakout_age", ()), "donchian_breakout_age")
    n = len(williams)
    if len(failure) != n or len(position) != n or len(age) != n:
        raise ValueError("Oscillator and Donchian state arrays must have the same length")
    williams = np.nan_to_num(williams, nan=0.0, posinf=0.0, neginf=0.0)
    failure = np.nan_to_num(failure, nan=0.0, posinf=0.0, neginf=0.0)
    position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
    age = np.nan_to_num(age, nan=0.0, posinf=0.0, neginf=0.0)
    failed_extension = -np.sign(position) * np.maximum(np.abs(position) - 0.72, 0.0)
    exhaustion = -0.45 * williams + 0.35 * failure + 0.45 * failed_extension * np.exp(-np.abs(age))
    signal = np.tanh(exhaustion)
    if clip > 0:
        signal = np.clip(signal, -float(clip), float(clip))
    return np.ascontiguousarray(signal, dtype=np.float64)


def build_oscillator_exhaustion_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated oscillator exhaustion indicator-state bundle."""
    features = compute_oscillator_exhaustion_state(high, low, close, period=period, lag=lag, use_numba=use_numba)
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"oscillator_exhaustion_state": True},
        spec_names=("oscillator_exhaustion_state",),
        source_columns=("high", "low", "close"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _macd_acceleration_state_reference(
    close: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    macd_z = np.full(n, np.nan, dtype=np.float64)
    ppo_z = np.full(n, np.nan, dtype=np.float64)
    trix_z = np.full(n, np.nan, dtype=np.float64)
    acceleration = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return macd_z, ppo_z, trix_z, acceleration
    ema_fast = _ema_series(close, fast_period)
    ema_slow = _ema_series(close, slow_period)
    macd = ema_fast - ema_slow
    signal = _ema_series(macd, signal_period)
    ppo = macd / np.maximum(np.abs(ema_slow), 1e-12)
    ema1 = _ema_series(close, signal_period)
    ema2 = _ema_series(ema1, signal_period)
    ema3 = _ema_series(ema2, signal_period)
    trix = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        trix[i] = (ema3[i] - ema3[i - 1]) / max(abs(ema3[i - 1]), 1e-12)

    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        returns[i] = (close[i] - close[i - 1]) / max(abs(close[i - 1]), 1e-12)

    raw_macd = np.zeros(n, dtype=np.float64)
    raw_ppo = np.zeros(n, dtype=np.float64)
    raw_trix = np.zeros(n, dtype=np.float64)
    raw_accel = np.zeros(n, dtype=np.float64)
    for i in range(n):
        scale = _rolling_return_scale(close, i, slow_period)
        price = max(abs(close[i]), 1e-12)
        raw_macd[i] = float(np.clip((macd[i] / price) / scale, -6.0, 6.0))
        raw_ppo[i] = float(np.clip(ppo[i] / scale, -6.0, 6.0))
        raw_trix[i] = float(np.clip(trix[i] / scale, -6.0, 6.0))
        hist = macd[i] - signal[i]
        prev_hist = macd[i - 1] - signal[i - 1] if i > 0 else hist
        raw_accel[i] = float(np.clip(((hist - prev_hist) / price) / scale, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        macd_z[i] = raw_macd[j]
        ppo_z[i] = raw_ppo[j]
        trix_z[i] = raw_trix[j]
        acceleration[i] = raw_accel[j]
    return macd_z, ppo_z, trix_z, acceleration


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _macd_acceleration_state_kernel(
        close: np.ndarray,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        macd_z = np.empty(n, dtype=np.float64)
        ppo_z = np.empty(n, dtype=np.float64)
        trix_z = np.empty(n, dtype=np.float64)
        acceleration = np.empty(n, dtype=np.float64)
        for i in range(n):
            macd_z[i] = np.nan
            ppo_z[i] = np.nan
            trix_z[i] = np.nan
            acceleration[i] = np.nan
        if n == 0:
            return macd_z, ppo_z, trix_z, acceleration

        ema_fast = np.empty(n, dtype=np.float64)
        ema_slow = np.empty(n, dtype=np.float64)
        macd = np.empty(n, dtype=np.float64)
        signal = np.empty(n, dtype=np.float64)
        ppo = np.empty(n, dtype=np.float64)
        ema1 = np.empty(n, dtype=np.float64)
        ema2 = np.empty(n, dtype=np.float64)
        ema3 = np.empty(n, dtype=np.float64)
        trix = np.zeros(n, dtype=np.float64)
        ema_fast[0] = close[0]
        ema_slow[0] = close[0]
        ema1[0] = close[0]
        ema2[0] = close[0]
        ema3[0] = close[0]
        alpha_fast = 2.0 / (fast_period + 1.0)
        alpha_slow = 2.0 / (slow_period + 1.0)
        alpha_signal = 2.0 / (signal_period + 1.0)
        for i in range(n):
            if i > 0:
                ema_fast[i] = alpha_fast * close[i] + (1.0 - alpha_fast) * ema_fast[i - 1]
                ema_slow[i] = alpha_slow * close[i] + (1.0 - alpha_slow) * ema_slow[i - 1]
                ema1[i] = alpha_signal * close[i] + (1.0 - alpha_signal) * ema1[i - 1]
                ema2[i] = alpha_signal * ema1[i] + (1.0 - alpha_signal) * ema2[i - 1]
                ema3[i] = alpha_signal * ema2[i] + (1.0 - alpha_signal) * ema3[i - 1]
                denom_e3 = abs(ema3[i - 1])
                if denom_e3 < 1e-12:
                    denom_e3 = 1e-12
                trix[i] = (ema3[i] - ema3[i - 1]) / denom_e3
            macd[i] = ema_fast[i] - ema_slow[i]
            denom_slow = abs(ema_slow[i])
            if denom_slow < 1e-12:
                denom_slow = 1e-12
            ppo[i] = macd[i] / denom_slow
            if i == 0:
                signal[i] = macd[i]
            else:
                signal[i] = alpha_signal * macd[i] + (1.0 - alpha_signal) * signal[i - 1]

        raw_macd = np.zeros(n, dtype=np.float64)
        raw_ppo = np.zeros(n, dtype=np.float64)
        raw_trix = np.zeros(n, dtype=np.float64)
        raw_accel = np.zeros(n, dtype=np.float64)
        for i in range(n):
            scale_start = i - slow_period + 1
            if scale_start < 1:
                scale_start = 1
            count = i - scale_start + 1
            scale = 1e-6
            if count > 0:
                mean = 0.0
                for j in range(scale_start, i + 1):
                    denom_prev = abs(close[j - 1])
                    if denom_prev < 1e-12:
                        denom_prev = 1e-12
                    mean += (close[j] - close[j - 1]) / denom_prev
                mean /= count
                var = 0.0
                for j in range(scale_start, i + 1):
                    denom_prev = abs(close[j - 1])
                    if denom_prev < 1e-12:
                        denom_prev = 1e-12
                    ret = (close[j] - close[j - 1]) / denom_prev
                    diff = ret - mean
                    var += diff * diff
                scale = np.sqrt(var / count)
                if scale < 1e-6:
                    scale = 1e-6
            price = abs(close[i])
            if price < 1e-12:
                price = 1e-12
            hist = macd[i] - signal[i]
            prev_hist = hist
            if i > 0:
                prev_hist = macd[i - 1] - signal[i - 1]
            mz = (macd[i] / price) / scale
            pz = ppo[i] / scale
            tz = trix[i] / scale
            az = ((hist - prev_hist) / price) / scale
            if mz < -6.0:
                mz = -6.0
            elif mz > 6.0:
                mz = 6.0
            if pz < -6.0:
                pz = -6.0
            elif pz > 6.0:
                pz = 6.0
            if tz < -6.0:
                tz = -6.0
            elif tz > 6.0:
                tz = 6.0
            if az < -6.0:
                az = -6.0
            elif az > 6.0:
                az = 6.0
            raw_macd[i] = mz
            raw_ppo[i] = pz
            raw_trix[i] = tz
            raw_accel[i] = az

        for i in range(lag, n):
            j = i - lag
            macd_z[i] = raw_macd[j]
            ppo_z[i] = raw_ppo[j]
            trix_z[i] = raw_trix[j]
            acceleration[i] = raw_accel[j]
        return macd_z, ppo_z, trix_z, acceleration
else:
    _macd_acceleration_state_kernel = None


def compute_macd_acceleration_state(
    close: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged MACD/PPO/TRIX acceleration state."""
    if not (1 <= fast_period < slow_period and signal_period >= 2):
        raise ValueError("MACD periods must satisfy 1 <= fast < slow and signal >= 2")
    if lag < 1:
        raise ValueError("MACD model state must be lagged by at least one bar")
    close_arr = _as_float_array(close, "close")
    if not bool(np.all(np.isfinite(close_arr))):
        raise ValueError("close array must be finite")
    if use_numba and _macd_acceleration_state_kernel is not None:
        values = _macd_acceleration_state_kernel(
            close_arr, int(fast_period), int(slow_period), int(signal_period), int(lag)
        )
    else:
        values = _macd_acceleration_state_reference(
            close_arr, int(fast_period), int(slow_period), int(signal_period), int(lag)
        )
    outputs = get_indicator_feature_spec("macd_acceleration_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def orthogonalize_momentum_features(
    feature_matrix: Sequence[Sequence[float]],
    eps: float = 1e-10,
) -> np.ndarray:
    """Return a stable Gram-Schmidt basis for momentum-like feature columns."""
    x = np.asarray(feature_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    q_cols = []
    for j in range(x.shape[1]):
        v = x[:, j].astype(np.float64).copy()
        v -= float(np.mean(v)) if v.size else 0.0
        for q in q_cols:
            denom = float(np.dot(q, q))
            if denom > eps:
                v -= q * (float(np.dot(v, q)) / denom)
        norm = float(np.sqrt(np.dot(v, v)))
        if norm > eps:
            q_cols.append(v / norm)
        else:
            q_cols.append(np.zeros_like(v))
    if not q_cols:
        return np.empty((x.shape[0], 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(q_cols), dtype=np.float64)


def build_macd_acceleration_bundle(
    close: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated MACD/PPO/TRIX indicator-state bundle."""
    features = compute_macd_acceleration_state(
        close,
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"macd_acceleration_state": True},
        spec_names=("macd_acceleration_state",),
        source_columns=("close",),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _volume_flow_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    obv_z = np.full(n, np.nan, dtype=np.float64)
    mfi_z = np.full(n, np.nan, dtype=np.float64)
    cmf_z = np.full(n, np.nan, dtype=np.float64)
    volume_z = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return obv_z, mfi_z, cmf_z, volume_z

    log_volume = np.log1p(volume)
    typical = (high + low + close) / 3.0
    raw_obv = np.zeros(n, dtype=np.float64)
    raw_mfi = np.zeros(n, dtype=np.float64)
    raw_cmf = np.zeros(n, dtype=np.float64)
    raw_volume = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        sum_signed_volume = 0.0
        sum_volume = 0.0
        positive_flow = 0.0
        negative_flow = 0.0
        cmf_num = 0.0
        for j in range(start, i + 1):
            vol = max(float(volume[j]), 0.0)
            sum_volume += vol
            price_sign = 0.0
            if j > 0:
                diff = close[j] - close[j - 1]
                if diff > 0.0:
                    price_sign = 1.0
                elif diff < 0.0:
                    price_sign = -1.0
            sum_signed_volume += price_sign * vol
            raw_money_flow = typical[j] * vol
            if j > 0 and typical[j] > typical[j - 1]:
                positive_flow += raw_money_flow
            elif j > 0 and typical[j] < typical[j - 1]:
                negative_flow += raw_money_flow
            bar_range = max(high[j] - low[j], 1e-12)
            mf_mult = ((close[j] - low[j]) - (high[j] - close[j])) / bar_range
            cmf_num += float(np.clip(mf_mult, -1.0, 1.0)) * vol

        denom_vol = max(sum_volume, 1e-12)
        raw_obv[i] = float(np.clip(sum_signed_volume / denom_vol, -1.0, 1.0))
        if positive_flow + negative_flow <= 1e-12:
            raw_mfi[i] = 0.0
        elif negative_flow <= 1e-12:
            raw_mfi[i] = 1.0
        elif positive_flow <= 1e-12:
            raw_mfi[i] = -1.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100.0 - (100.0 / (1.0 + money_ratio))
            raw_mfi[i] = float(np.clip((mfi - 50.0) / 50.0, -1.0, 1.0))
        raw_cmf[i] = float(np.clip(cmf_num / denom_vol, -1.0, 1.0))
        vol_window = log_volume[start : i + 1]
        vol_std = max(float(np.std(vol_window)), 1e-6)
        raw_volume[i] = float(np.clip((log_volume[i] - float(np.mean(vol_window))) / vol_std, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        obv_z[i] = raw_obv[j]
        mfi_z[i] = raw_mfi[j]
        cmf_z[i] = raw_cmf[j]
        volume_z[i] = raw_volume[j]
    return obv_z, mfi_z, cmf_z, volume_z


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _volume_flow_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        obv_z = np.empty(n, dtype=np.float64)
        mfi_z = np.empty(n, dtype=np.float64)
        cmf_z = np.empty(n, dtype=np.float64)
        volume_z = np.empty(n, dtype=np.float64)
        for i in range(n):
            obv_z[i] = np.nan
            mfi_z[i] = np.nan
            cmf_z[i] = np.nan
            volume_z[i] = np.nan
        if n == 0:
            return obv_z, mfi_z, cmf_z, volume_z

        log_volume = np.empty(n, dtype=np.float64)
        typical = np.empty(n, dtype=np.float64)
        raw_obv = np.zeros(n, dtype=np.float64)
        raw_mfi = np.zeros(n, dtype=np.float64)
        raw_cmf = np.zeros(n, dtype=np.float64)
        raw_volume = np.zeros(n, dtype=np.float64)
        for i in range(n):
            log_volume[i] = np.log1p(volume[i])
            typical[i] = (high[i] + low[i] + close[i]) / 3.0

        for i in range(n):
            start = i - period + 1
            if start < 0:
                start = 0
            sum_signed_volume = 0.0
            sum_volume = 0.0
            positive_flow = 0.0
            negative_flow = 0.0
            cmf_num = 0.0
            for j in range(start, i + 1):
                vol = volume[j]
                if vol < 0.0:
                    vol = 0.0
                sum_volume += vol
                price_sign = 0.0
                if j > 0:
                    diff = close[j] - close[j - 1]
                    if diff > 0.0:
                        price_sign = 1.0
                    elif diff < 0.0:
                        price_sign = -1.0
                sum_signed_volume += price_sign * vol
                raw_money_flow = typical[j] * vol
                if j > 0 and typical[j] > typical[j - 1]:
                    positive_flow += raw_money_flow
                elif j > 0 and typical[j] < typical[j - 1]:
                    negative_flow += raw_money_flow
                bar_range = high[j] - low[j]
                if bar_range < 1e-12:
                    bar_range = 1e-12
                mf_mult = ((close[j] - low[j]) - (high[j] - close[j])) / bar_range
                if mf_mult < -1.0:
                    mf_mult = -1.0
                elif mf_mult > 1.0:
                    mf_mult = 1.0
                cmf_num += mf_mult * vol

            denom_vol = sum_volume
            if denom_vol < 1e-12:
                denom_vol = 1e-12
            obv = sum_signed_volume / denom_vol
            if obv < -1.0:
                obv = -1.0
            elif obv > 1.0:
                obv = 1.0
            raw_obv[i] = obv
            if positive_flow + negative_flow <= 1e-12:
                raw_mfi[i] = 0.0
            elif negative_flow <= 1e-12:
                raw_mfi[i] = 1.0
            elif positive_flow <= 1e-12:
                raw_mfi[i] = -1.0
            else:
                money_ratio = positive_flow / negative_flow
                mfi = 100.0 - (100.0 / (1.0 + money_ratio))
                value = (mfi - 50.0) / 50.0
                if value < -1.0:
                    value = -1.0
                elif value > 1.0:
                    value = 1.0
                raw_mfi[i] = value
            cmf = cmf_num / denom_vol
            if cmf < -1.0:
                cmf = -1.0
            elif cmf > 1.0:
                cmf = 1.0
            raw_cmf[i] = cmf

            count = i - start + 1
            mean = 0.0
            for j in range(start, i + 1):
                mean += log_volume[j]
            mean /= count
            var = 0.0
            for j in range(start, i + 1):
                diff = log_volume[j] - mean
                var += diff * diff
            vol_std = np.sqrt(var / count)
            if vol_std < 1e-6:
                vol_std = 1e-6
            vz = (log_volume[i] - mean) / vol_std
            if vz < -6.0:
                vz = -6.0
            elif vz > 6.0:
                vz = 6.0
            raw_volume[i] = vz

        for i in range(lag, n):
            j = i - lag
            obv_z[i] = raw_obv[j]
            mfi_z[i] = raw_mfi[j]
            cmf_z[i] = raw_cmf[j]
            volume_z[i] = raw_volume[j]
        return obv_z, mfi_z, cmf_z, volume_z
else:
    _volume_flow_state_kernel = None


def compute_volume_flow_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 21,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged OBV/MFI/CMF/volume state for model conditioning."""
    if period < 2:
        raise ValueError("volume flow period must be at least 2")
    if lag < 1:
        raise ValueError("volume flow model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    volume_arr = _as_float_array(volume, "volume")
    _validate_hlcv_arrays(high_arr, low_arr, close_arr, volume_arr)
    if use_numba and _volume_flow_state_kernel is not None:
        values = _volume_flow_state_kernel(high_arr, low_arr, close_arr, volume_arr, int(period), int(lag))
    else:
        values = _volume_flow_state_reference(high_arr, low_arr, close_arr, volume_arr, int(period), int(lag))
    outputs = get_indicator_feature_spec("volume_flow_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_volume_flow_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 21,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated OBV/MFI/CMF volume-flow state bundle."""
    features = compute_volume_flow_state(
        high, low, close, volume, period=period, lag=lag, use_numba=use_numba
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"volume_flow_state": True},
        spec_names=("volume_flow_state",),
        source_columns=("high", "low", "close", "volume"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def _rolling_log_dollar_volume_z(
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
    lag: int,
) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    dollar_volume = np.log1p(np.maximum(np.abs(close) * volume, 0.0))
    raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        window = dollar_volume[start : i + 1]
        scale = max(float(np.std(window)), 1e-6)
        raw[i] = float(np.clip((dollar_volume[i] - float(np.mean(window))) / scale, -6.0, 6.0))
    for i in range(lag, n):
        out[i] = raw[i - lag]
    return out


def compute_liquidity_variance_conditioner(
    close: Sequence[float],
    volume: Sequence[float],
    volume_flow_state: Optional[Mapping[str, np.ndarray]] = None,
    period: int = 21,
    lag: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bounded variance and confidence multipliers from liquidity state."""
    if period < 2:
        raise ValueError("liquidity period must be at least 2")
    if lag < 1:
        raise ValueError("liquidity model state must be lagged by at least one bar")
    close_arr = _as_float_array(close, "close")
    volume_arr = _as_float_array(volume, "volume")
    if len(close_arr) != len(volume_arr):
        raise ValueError("close and volume arrays must have the same length")
    if not bool(np.all(np.isfinite(close_arr) & np.isfinite(volume_arr))):
        raise ValueError("close and volume arrays must be finite")
    if not bool(np.all(volume_arr >= 0.0)):
        raise ValueError("volume must be non-negative")
    n = len(close_arr)
    if volume_flow_state is not None and "volume_z" in volume_flow_state:
        volume_z = np.asarray(volume_flow_state["volume_z"], dtype=np.float64)
        if len(volume_z) != n:
            raise ValueError("volume_z length must match close")
        volume_z = np.nan_to_num(volume_z, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        volume_z = _rolling_log_dollar_volume_z(np.ones(n, dtype=np.float64), volume_arr, period, lag)
        volume_z = np.nan_to_num(volume_z, nan=0.0, posinf=0.0, neginf=0.0)
    dollar_z = _rolling_log_dollar_volume_z(close_arr, volume_arr, period, lag)
    dollar_z = np.nan_to_num(dollar_z, nan=0.0, posinf=0.0, neginf=0.0)

    liquidity_deficit = np.clip(-dollar_z, 0.0, 3.0) / 3.0
    liquidity_surplus = np.clip(dollar_z, 0.0, 2.0) / 2.0
    volume_abnormal = np.clip(np.abs(volume_z) - 1.0, 0.0, 3.0) / 3.0
    volume_dry_up = np.clip(-volume_z, 0.0, 3.0) / 3.0

    variance_mult = 1.0 + 0.35 * liquidity_deficit + 0.25 * volume_abnormal + 0.15 * volume_dry_up
    variance_mult -= 0.10 * liquidity_surplus * (1.0 - np.clip(np.abs(volume_z) / 3.0, 0.0, 1.0))
    confidence_mult = 1.0 + 0.25 * liquidity_surplus
    confidence_mult -= 0.35 * liquidity_deficit + 0.20 * volume_abnormal
    variance_mult = np.clip(variance_mult, 0.85, 1.60)
    confidence_mult = np.clip(confidence_mult, 0.70, 1.25)
    if n:
        variance_mult[:lag] = 1.0
        confidence_mult[:lag] = 1.0
    return (
        np.ascontiguousarray(variance_mult, dtype=np.float64),
        np.ascontiguousarray(confidence_mult, dtype=np.float64),
    )


def _vwap_dislocation_state_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int,
    anchor_period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    vwap_distance = np.full(n, np.nan, dtype=np.float64)
    vwap_band_z = np.full(n, np.nan, dtype=np.float64)
    vwap_slope = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return vwap_distance, vwap_band_z, vwap_slope

    typical = (high + low + close) / 3.0
    raw_distance = np.zeros(n, dtype=np.float64)
    raw_band = np.zeros(n, dtype=np.float64)
    raw_slope = np.zeros(n, dtype=np.float64)
    blended = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        anchor_start = (i // anchor_period) * anchor_period
        rolling_num = 0.0
        rolling_den = 0.0
        rolling_equal = 0.0
        for j in range(start, i + 1):
            vol = max(float(volume[j]), 0.0)
            rolling_num += typical[j] * vol
            rolling_den += vol
            rolling_equal += typical[j]
        count = i - start + 1
        if rolling_den > 1e-12:
            rolling_vwap = rolling_num / rolling_den
        else:
            rolling_vwap = rolling_equal / max(count, 1)

        anchor_num = 0.0
        anchor_den = 0.0
        anchor_equal = 0.0
        for j in range(anchor_start, i + 1):
            vol = max(float(volume[j]), 0.0)
            anchor_num += typical[j] * vol
            anchor_den += vol
            anchor_equal += typical[j]
        anchor_count = i - anchor_start + 1
        if anchor_den > 1e-12:
            anchor_vwap = anchor_num / anchor_den
        else:
            anchor_vwap = anchor_equal / max(anchor_count, 1)

        blended[i] = 0.65 * rolling_vwap + 0.35 * anchor_vwap
        var_num = 0.0
        var_den = 0.0
        equal_var = 0.0
        for j in range(start, i + 1):
            diff = typical[j] - rolling_vwap
            vol = max(float(volume[j]), 0.0)
            var_num += vol * diff * diff
            var_den += vol
            equal_var += diff * diff
        if var_den > 1e-12:
            band_scale = float(np.sqrt(var_num / var_den))
        else:
            band_scale = float(np.sqrt(equal_var / max(count, 1)))
        band_scale = max(band_scale, abs(close[i]) * 1e-6, 1e-12)
        raw_distance[i] = float(np.clip((close[i] - blended[i]) / band_scale, -6.0, 6.0))
        raw_band[i] = float(np.clip((typical[i] - rolling_vwap) / band_scale, -6.0, 6.0))
        if i == 0:
            raw_slope[i] = 0.0
        else:
            denom = max(abs(blended[i - 1]), 1e-12)
            scale = _rolling_return_scale(close, i, period)
            raw_slope[i] = float(np.clip(((blended[i] - blended[i - 1]) / denom) / scale, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        vwap_distance[i] = raw_distance[j]
        vwap_band_z[i] = raw_band[j]
        vwap_slope[i] = raw_slope[j]
    return vwap_distance, vwap_band_z, vwap_slope


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _vwap_dislocation_state_kernel(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int,
        anchor_period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        vwap_distance = np.empty(n, dtype=np.float64)
        vwap_band_z = np.empty(n, dtype=np.float64)
        vwap_slope = np.empty(n, dtype=np.float64)
        for i in range(n):
            vwap_distance[i] = np.nan
            vwap_band_z[i] = np.nan
            vwap_slope[i] = np.nan
        if n == 0:
            return vwap_distance, vwap_band_z, vwap_slope

        typical = np.empty(n, dtype=np.float64)
        raw_distance = np.zeros(n, dtype=np.float64)
        raw_band = np.zeros(n, dtype=np.float64)
        raw_slope = np.zeros(n, dtype=np.float64)
        blended = np.zeros(n, dtype=np.float64)
        for i in range(n):
            typical[i] = (high[i] + low[i] + close[i]) / 3.0

        for i in range(n):
            start = i - period + 1
            if start < 0:
                start = 0
            anchor_start = (i // anchor_period) * anchor_period
            rolling_num = 0.0
            rolling_den = 0.0
            rolling_equal = 0.0
            for j in range(start, i + 1):
                vol = volume[j]
                if vol < 0.0:
                    vol = 0.0
                rolling_num += typical[j] * vol
                rolling_den += vol
                rolling_equal += typical[j]
            count = i - start + 1
            rolling_vwap = rolling_equal / count
            if rolling_den > 1e-12:
                rolling_vwap = rolling_num / rolling_den

            anchor_num = 0.0
            anchor_den = 0.0
            anchor_equal = 0.0
            for j in range(anchor_start, i + 1):
                vol = volume[j]
                if vol < 0.0:
                    vol = 0.0
                anchor_num += typical[j] * vol
                anchor_den += vol
                anchor_equal += typical[j]
            anchor_count = i - anchor_start + 1
            anchor_vwap = anchor_equal / anchor_count
            if anchor_den > 1e-12:
                anchor_vwap = anchor_num / anchor_den

            blended[i] = 0.65 * rolling_vwap + 0.35 * anchor_vwap
            var_num = 0.0
            var_den = 0.0
            equal_var = 0.0
            for j in range(start, i + 1):
                diff = typical[j] - rolling_vwap
                vol = volume[j]
                if vol < 0.0:
                    vol = 0.0
                var_num += vol * diff * diff
                var_den += vol
                equal_var += diff * diff
            if var_den > 1e-12:
                band_scale = np.sqrt(var_num / var_den)
            else:
                band_scale = np.sqrt(equal_var / count)
            min_scale = abs(close[i]) * 1e-6
            if band_scale < min_scale:
                band_scale = min_scale
            if band_scale < 1e-12:
                band_scale = 1e-12
            dz = (close[i] - blended[i]) / band_scale
            bz = (typical[i] - rolling_vwap) / band_scale
            if dz < -6.0:
                dz = -6.0
            elif dz > 6.0:
                dz = 6.0
            if bz < -6.0:
                bz = -6.0
            elif bz > 6.0:
                bz = 6.0
            raw_distance[i] = dz
            raw_band[i] = bz
            if i == 0:
                raw_slope[i] = 0.0
            else:
                denom = abs(blended[i - 1])
                if denom < 1e-12:
                    denom = 1e-12
                scale_start = i - period + 1
                if scale_start < 1:
                    scale_start = 1
                ret_count = i - scale_start + 1
                scale = 1e-6
                if ret_count > 0:
                    mean = 0.0
                    for j in range(scale_start, i + 1):
                        prev = abs(close[j - 1])
                        if prev < 1e-12:
                            prev = 1e-12
                        mean += (close[j] - close[j - 1]) / prev
                    mean /= ret_count
                    var = 0.0
                    for j in range(scale_start, i + 1):
                        prev = abs(close[j - 1])
                        if prev < 1e-12:
                            prev = 1e-12
                        ret = (close[j] - close[j - 1]) / prev
                        diff_ret = ret - mean
                        var += diff_ret * diff_ret
                    scale = np.sqrt(var / ret_count)
                    if scale < 1e-6:
                        scale = 1e-6
                sz = ((blended[i] - blended[i - 1]) / denom) / scale
                if sz < -6.0:
                    sz = -6.0
                elif sz > 6.0:
                    sz = 6.0
                raw_slope[i] = sz

        for i in range(lag, n):
            j = i - lag
            vwap_distance[i] = raw_distance[j]
            vwap_band_z[i] = raw_band[j]
            vwap_slope[i] = raw_slope[j]
        return vwap_distance, vwap_band_z, vwap_slope
else:
    _vwap_dislocation_state_kernel = None


def compute_vwap_dislocation_state(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 63,
    anchor_period: int = 63,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged rolling/anchored VWAP equilibrium state."""
    if period < 2:
        raise ValueError("VWAP period must be at least 2")
    if anchor_period < 2:
        raise ValueError("VWAP anchor_period must be at least 2")
    if lag < 1:
        raise ValueError("VWAP model state must be lagged by at least one bar")
    high_arr = _as_float_array(high, "high")
    low_arr = _as_float_array(low, "low")
    close_arr = _as_float_array(close, "close")
    volume_arr = _as_float_array(volume, "volume")
    _validate_hlcv_arrays(high_arr, low_arr, close_arr, volume_arr)
    if use_numba and _vwap_dislocation_state_kernel is not None:
        values = _vwap_dislocation_state_kernel(
            high_arr, low_arr, close_arr, volume_arr, int(period), int(anchor_period), int(lag)
        )
    else:
        values = _vwap_dislocation_state_reference(
            high_arr, low_arr, close_arr, volume_arr, int(period), int(anchor_period), int(lag)
        )
    outputs = get_indicator_feature_spec("vwap_dislocation_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_vwap_dislocation_bundle(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int = 63,
    anchor_period: int = 63,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated rolling/anchored VWAP model-state bundle."""
    features = compute_vwap_dislocation_state(
        high,
        low,
        close,
        volume,
        period=period,
        anchor_period=anchor_period,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"vwap_dislocation_state": True},
        spec_names=("vwap_dislocation_state",),
        source_columns=("high", "low", "close", "volume"),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def compute_vwap_confidence_conditioner(
    vwap_state: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bounded VWAP-dislocation variance and confidence multipliers."""
    distance = np.nan_to_num(np.asarray(vwap_state["vwap_distance"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    band = np.nan_to_num(np.asarray(vwap_state["vwap_band_z"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    slope = np.nan_to_num(np.asarray(vwap_state["vwap_slope"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(distance) != len(band) or len(distance) != len(slope):
        raise ValueError("VWAP state arrays must have the same length")
    dislocation = np.clip(np.abs(distance) / 3.0, 0.0, 1.0)
    band_pressure = np.clip(np.abs(band) / 3.0, 0.0, 1.0)
    slope_alignment = np.clip((np.sign(distance) * np.sign(slope) + 1.0) * 0.5, 0.0, 1.0)
    variance_mult = 1.0 + 0.35 * dislocation + 0.15 * band_pressure + 0.10 * dislocation * slope_alignment
    confidence_mult = 1.0 - 0.30 * dislocation + 0.12 * slope_alignment * (1.0 - dislocation)
    variance_mult = np.clip(variance_mult, 0.90, 1.60)
    confidence_mult = np.clip(confidence_mult, 0.70, 1.20)
    if len(distance):
        variance_mult[0] = 1.0
        confidence_mult[0] = 1.0
    return (
        np.ascontiguousarray(variance_mult, dtype=np.float64),
        np.ascontiguousarray(confidence_mult, dtype=np.float64),
    )


def _persistence_state_reference(
    close: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    hurst_proxy = np.full(n, np.nan, dtype=np.float64)
    fractal_dimension = np.full(n, np.nan, dtype=np.float64)
    wavelet_energy_z = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return hurst_proxy, fractal_dimension, wavelet_energy_z

    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        returns[i] = (close[i] - close[i - 1]) / max(abs(close[i - 1]), 1e-12)

    raw_hurst = np.full(n, 0.5, dtype=np.float64)
    raw_fractal = np.full(n, 1.5, dtype=np.float64)
    raw_energy = np.zeros(n, dtype=np.float64)
    lags = (2, 4, 8)
    for i in range(n):
        start = max(1, i - period + 1)
        count = i - start + 1
        if count <= 2:
            raw_hurst[i] = 0.5
            raw_fractal[i] = 1.5
            raw_energy[i] = 0.0
            continue
        ret_window = returns[start : i + 1]
        var1 = max(float(np.var(ret_window)), 1e-12)
        h_sum = 0.0
        h_count = 0
        for step in lags:
            if count <= step:
                continue
            stepped = []
            for j in range(start + step - 1, i + 1):
                stepped.append(float(np.sum(returns[j - step + 1 : j + 1])))
            var_step = max(float(np.var(np.asarray(stepped, dtype=np.float64))), 1e-12)
            h_val = 0.5 * (np.log(var_step / var1) / np.log(float(step)))
            h_sum += float(np.clip(h_val, 0.0, 1.0))
            h_count += 1
        h = 0.5 if h_count == 0 else h_sum / h_count
        raw_hurst[i] = float(np.clip(h, 0.0, 1.0))
        raw_fractal[i] = float(np.clip(2.0 - raw_hurst[i], 1.0, 2.0))
        energy_start = max(start + 1, i - max(2, period // 4) + 1)
        diff_energy = 0.0
        diff_count = 0
        for j in range(energy_start, i + 1):
            diff = returns[j] - returns[j - 1]
            diff_energy += diff * diff
            diff_count += 1
        raw_energy[i] = diff_energy / max(diff_count, 1) / var1

    raw_energy_z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        window = raw_energy[start : i + 1]
        scale = max(float(np.std(window)), 1e-6)
        raw_energy_z[i] = float(np.clip((raw_energy[i] - float(np.mean(window))) / scale, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        hurst_proxy[i] = raw_hurst[j]
        fractal_dimension[i] = raw_fractal[j]
        wavelet_energy_z[i] = raw_energy_z[j]
    return hurst_proxy, fractal_dimension, wavelet_energy_z


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _persistence_state_kernel(
        close: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        hurst_proxy = np.empty(n, dtype=np.float64)
        fractal_dimension = np.empty(n, dtype=np.float64)
        wavelet_energy_z = np.empty(n, dtype=np.float64)
        for i in range(n):
            hurst_proxy[i] = np.nan
            fractal_dimension[i] = np.nan
            wavelet_energy_z[i] = np.nan
        if n == 0:
            return hurst_proxy, fractal_dimension, wavelet_energy_z

        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            denom = abs(close[i - 1])
            if denom < 1e-12:
                denom = 1e-12
            returns[i] = (close[i] - close[i - 1]) / denom

        raw_hurst = np.empty(n, dtype=np.float64)
        raw_fractal = np.empty(n, dtype=np.float64)
        raw_energy = np.zeros(n, dtype=np.float64)
        for i in range(n):
            raw_hurst[i] = 0.5
            raw_fractal[i] = 1.5

        for i in range(n):
            start = i - period + 1
            if start < 1:
                start = 1
            count = i - start + 1
            if count <= 2:
                raw_hurst[i] = 0.5
                raw_fractal[i] = 1.5
                raw_energy[i] = 0.0
                continue
            mean1 = 0.0
            for j in range(start, i + 1):
                mean1 += returns[j]
            mean1 /= count
            var1 = 0.0
            for j in range(start, i + 1):
                diff = returns[j] - mean1
                var1 += diff * diff
            var1 /= count
            if var1 < 1e-12:
                var1 = 1e-12

            h_sum = 0.0
            h_count = 0
            for step_idx in range(3):
                step = 2
                if step_idx == 1:
                    step = 4
                elif step_idx == 2:
                    step = 8
                if count <= step:
                    continue
                step_count = i - (start + step - 1) + 1
                mean_step = 0.0
                for j in range(start + step - 1, i + 1):
                    accum = 0.0
                    for k in range(j - step + 1, j + 1):
                        accum += returns[k]
                    mean_step += accum
                mean_step /= step_count
                var_step = 0.0
                for j in range(start + step - 1, i + 1):
                    accum = 0.0
                    for k in range(j - step + 1, j + 1):
                        accum += returns[k]
                    diff_step = accum - mean_step
                    var_step += diff_step * diff_step
                var_step /= step_count
                if var_step < 1e-12:
                    var_step = 1e-12
                h_val = 0.5 * (np.log(var_step / var1) / np.log(float(step)))
                if h_val < 0.0:
                    h_val = 0.0
                elif h_val > 1.0:
                    h_val = 1.0
                h_sum += h_val
                h_count += 1
            h = 0.5
            if h_count > 0:
                h = h_sum / h_count
            if h < 0.0:
                h = 0.0
            elif h > 1.0:
                h = 1.0
            raw_hurst[i] = h
            fd = 2.0 - h
            if fd < 1.0:
                fd = 1.0
            elif fd > 2.0:
                fd = 2.0
            raw_fractal[i] = fd
            short_window = period // 4
            if short_window < 2:
                short_window = 2
            energy_start = i - short_window + 1
            if energy_start < start + 1:
                energy_start = start + 1
            diff_energy = 0.0
            diff_count = 0
            for j in range(energy_start, i + 1):
                diff_e = returns[j] - returns[j - 1]
                diff_energy += diff_e * diff_e
                diff_count += 1
            if diff_count <= 0:
                diff_count = 1
            raw_energy[i] = (diff_energy / diff_count) / var1

        raw_energy_z = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = i - period + 1
            if start < 0:
                start = 0
            count = i - start + 1
            mean = 0.0
            for j in range(start, i + 1):
                mean += raw_energy[j]
            mean /= count
            var = 0.0
            for j in range(start, i + 1):
                diff = raw_energy[j] - mean
                var += diff * diff
            scale = np.sqrt(var / count)
            if scale < 1e-6:
                scale = 1e-6
            ez = (raw_energy[i] - mean) / scale
            if ez < -6.0:
                ez = -6.0
            elif ez > 6.0:
                ez = 6.0
            raw_energy_z[i] = ez

        for i in range(lag, n):
            j = i - lag
            hurst_proxy[i] = raw_hurst[j]
            fractal_dimension[i] = raw_fractal[j]
            wavelet_energy_z[i] = raw_energy_z[j]
        return hurst_proxy, fractal_dimension, wavelet_energy_z
else:
    _persistence_state_kernel = None


def compute_persistence_state(
    close: Sequence[float],
    period: int = 126,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged Hurst/fractal/rough-energy state."""
    if period < 8:
        raise ValueError("persistence period must be at least 8")
    if lag < 1:
        raise ValueError("persistence model state must be lagged by at least one bar")
    close_arr = _as_float_array(close, "close")
    if not bool(np.all(np.isfinite(close_arr))):
        raise ValueError("close array must be finite")
    if use_numba and _persistence_state_kernel is not None:
        values = _persistence_state_kernel(close_arr, int(period), int(lag))
    else:
        values = _persistence_state_reference(close_arr, int(period), int(lag))
    outputs = get_indicator_feature_spec("persistence_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_persistence_bundle(
    close: Sequence[float],
    period: int = 126,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated Hurst/fractal/rough-energy state bundle."""
    features = compute_persistence_state(close, period=period, lag=lag, use_numba=use_numba)
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"persistence_state": True},
        spec_names=("persistence_state",),
        source_columns=("close",),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def compute_persistence_q_regime_conditioner(
    persistence_state: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bounded q and regime-fit multipliers from persistence state."""
    hurst = np.nan_to_num(np.asarray(persistence_state["hurst_proxy"], dtype=np.float64), nan=0.5, posinf=0.5, neginf=0.5)
    fractal = np.nan_to_num(
        np.asarray(persistence_state["fractal_dimension_proxy"], dtype=np.float64),
        nan=1.5,
        posinf=1.5,
        neginf=1.5,
    )
    energy = np.nan_to_num(np.asarray(persistence_state["wavelet_energy_z"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(hurst) != len(fractal) or len(hurst) != len(energy):
        raise ValueError("persistence state arrays must have the same length")
    structure = np.clip(np.abs(hurst - 0.5) * 2.0, 0.0, 1.0)
    roughness = np.clip((fractal - 1.5) * 2.0, 0.0, 1.0)
    energy_pressure = np.clip(energy, 0.0, 4.0) / 4.0
    q_mult = 1.0 + 0.35 * roughness + 0.35 * energy_pressure - 0.20 * structure
    regime_fit = 1.0 + 0.25 * structure - 0.20 * energy_pressure
    q_mult = np.clip(q_mult, 0.75, 1.55)
    regime_fit = np.clip(regime_fit, 0.75, 1.25)
    if len(hurst):
        q_mult[0] = 1.0
        regime_fit[0] = 1.0
    return (
        np.ascontiguousarray(q_mult, dtype=np.float64),
        np.ascontiguousarray(regime_fit, dtype=np.float64),
    )


def compute_wavelet_energy_tail_conditioner(
    persistence_state: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return bounded variance and tail multipliers from rough-energy shocks."""
    energy = np.nan_to_num(np.asarray(persistence_state["wavelet_energy_z"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pressure = np.clip(energy, 0.0, 4.0) / 4.0
    relief = np.clip(-energy, 0.0, 3.0) / 3.0
    variance_mult = 1.0 + 0.55 * pressure - 0.08 * relief
    tail_mult = 1.0 + 0.30 * pressure
    variance_mult = np.clip(variance_mult, 0.90, 1.70)
    tail_mult = np.clip(tail_mult, 0.95, 1.35)
    if len(energy):
        variance_mult[0] = 1.0
        tail_mult[0] = 1.0
    return (
        np.ascontiguousarray(variance_mult, dtype=np.float64),
        np.ascontiguousarray(tail_mult, dtype=np.float64),
    )


def _as_peer_matrix(values: Optional[Sequence[Sequence[float]]], n_obs: int) -> np.ndarray:
    if values is None:
        return np.empty((n_obs, 0), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(n_obs, 1)
    if arr.ndim != 2:
        raise ValueError("peer_close_matrix must be 2D")
    if arr.shape[0] != n_obs:
        raise ValueError("peer_close_matrix row count must match close")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("peer_close_matrix must be finite")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _relative_strength_state_reference(
    close: np.ndarray,
    benchmark_close: np.ndarray,
    peer_close_matrix: np.ndarray,
    period: int,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    relative_strength_z = np.full(n, np.nan, dtype=np.float64)
    breadth_context = np.full(n, np.nan, dtype=np.float64)
    beta_adjusted_momentum = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return relative_strength_z, breadth_context, beta_adjusted_momentum
    m = peer_close_matrix.shape[1]
    raw_relative = np.zeros(n, dtype=np.float64)
    raw_breadth = np.zeros(n, dtype=np.float64)
    raw_beta_adjusted = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        asset_base = max(abs(close[start]), 1e-12)
        bench_base = max(abs(benchmark_close[start]), 1e-12)
        asset_mom = close[i] / asset_base - 1.0
        bench_mom = benchmark_close[i] / bench_base - 1.0
        peer_moms = np.zeros(m, dtype=np.float64)
        positive_peers = 0
        for j in range(m):
            peer_base = max(abs(peer_close_matrix[start, j]), 1e-12)
            mom = peer_close_matrix[i, j] / peer_base - 1.0
            peer_moms[j] = mom
            if mom > 0.0:
                positive_peers += 1
        if m:
            peer_mean = float(np.mean(peer_moms))
            peer_std = max(float(np.std(peer_moms)), 1e-6)
            context_mom = 0.50 * bench_mom + 0.50 * peer_mean
            raw_breadth[i] = 2.0 * (positive_peers / float(m)) - 1.0
        else:
            peer_std = max(abs(bench_mom), 1e-6)
            context_mom = bench_mom
            raw_breadth[i] = 1.0 if bench_mom > 0.0 else -1.0 if bench_mom < 0.0 else 0.0
        raw_relative[i] = float(np.clip((asset_mom - context_mom) / peer_std, -6.0, 6.0))

        ret_start = max(1, start + 1)
        ret_count = i - ret_start + 1
        beta = 1.0
        if ret_count >= 2:
            asset_returns = np.zeros(ret_count, dtype=np.float64)
            bench_returns = np.zeros(ret_count, dtype=np.float64)
            for k, t in enumerate(range(ret_start, i + 1)):
                asset_returns[k] = (close[t] - close[t - 1]) / max(abs(close[t - 1]), 1e-12)
                bench_returns[k] = (benchmark_close[t] - benchmark_close[t - 1]) / max(abs(benchmark_close[t - 1]), 1e-12)
            bench_var = float(np.var(bench_returns))
            if bench_var > 1e-12:
                beta = float(np.cov(asset_returns, bench_returns, bias=True)[0, 1] / bench_var)
                beta = float(np.clip(beta, -3.0, 3.0))
        raw_beta_adjusted[i] = float(np.clip((asset_mom - beta * bench_mom) / peer_std, -6.0, 6.0))

    for i in range(lag, n):
        j = i - lag
        relative_strength_z[i] = raw_relative[j]
        breadth_context[i] = raw_breadth[j]
        beta_adjusted_momentum[i] = raw_beta_adjusted[j]
    return relative_strength_z, breadth_context, beta_adjusted_momentum


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _relative_strength_state_kernel(
        close: np.ndarray,
        benchmark_close: np.ndarray,
        peer_close_matrix: np.ndarray,
        period: int,
        lag: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(close)
        relative_strength_z = np.empty(n, dtype=np.float64)
        breadth_context = np.empty(n, dtype=np.float64)
        beta_adjusted_momentum = np.empty(n, dtype=np.float64)
        for i in range(n):
            relative_strength_z[i] = np.nan
            breadth_context[i] = np.nan
            beta_adjusted_momentum[i] = np.nan
        if n == 0:
            return relative_strength_z, breadth_context, beta_adjusted_momentum
        m = peer_close_matrix.shape[1]
        raw_relative = np.zeros(n, dtype=np.float64)
        raw_breadth = np.zeros(n, dtype=np.float64)
        raw_beta_adjusted = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = i - period + 1
            if start < 0:
                start = 0
            asset_base = abs(close[start])
            if asset_base < 1e-12:
                asset_base = 1e-12
            bench_base = abs(benchmark_close[start])
            if bench_base < 1e-12:
                bench_base = 1e-12
            asset_mom = close[i] / asset_base - 1.0
            bench_mom = benchmark_close[i] / bench_base - 1.0
            peer_sum = 0.0
            peer_sq = 0.0
            positive_peers = 0
            for j in range(m):
                peer_base = abs(peer_close_matrix[start, j])
                if peer_base < 1e-12:
                    peer_base = 1e-12
                mom = peer_close_matrix[i, j] / peer_base - 1.0
                peer_sum += mom
                peer_sq += mom * mom
                if mom > 0.0:
                    positive_peers += 1
            peer_std = 1e-6
            context_mom = bench_mom
            if m > 0:
                peer_mean = peer_sum / m
                var_peer = peer_sq / m - peer_mean * peer_mean
                if var_peer < 0.0:
                    var_peer = 0.0
                peer_std = np.sqrt(var_peer)
                if peer_std < 1e-6:
                    peer_std = 1e-6
                context_mom = 0.50 * bench_mom + 0.50 * peer_mean
                raw_breadth[i] = 2.0 * (positive_peers / float(m)) - 1.0
            else:
                peer_std = abs(bench_mom)
                if peer_std < 1e-6:
                    peer_std = 1e-6
                if bench_mom > 0.0:
                    raw_breadth[i] = 1.0
                elif bench_mom < 0.0:
                    raw_breadth[i] = -1.0
                else:
                    raw_breadth[i] = 0.0
            rz = (asset_mom - context_mom) / peer_std
            if rz < -6.0:
                rz = -6.0
            elif rz > 6.0:
                rz = 6.0
            raw_relative[i] = rz

            ret_start = start + 1
            if ret_start < 1:
                ret_start = 1
            ret_count = i - ret_start + 1
            beta = 1.0
            if ret_count >= 2:
                mean_asset = 0.0
                mean_bench = 0.0
                for t in range(ret_start, i + 1):
                    prev_asset = abs(close[t - 1])
                    if prev_asset < 1e-12:
                        prev_asset = 1e-12
                    prev_bench = abs(benchmark_close[t - 1])
                    if prev_bench < 1e-12:
                        prev_bench = 1e-12
                    mean_asset += (close[t] - close[t - 1]) / prev_asset
                    mean_bench += (benchmark_close[t] - benchmark_close[t - 1]) / prev_bench
                mean_asset /= ret_count
                mean_bench /= ret_count
                cov = 0.0
                var_bench = 0.0
                for t in range(ret_start, i + 1):
                    prev_asset = abs(close[t - 1])
                    if prev_asset < 1e-12:
                        prev_asset = 1e-12
                    prev_bench = abs(benchmark_close[t - 1])
                    if prev_bench < 1e-12:
                        prev_bench = 1e-12
                    ar = (close[t] - close[t - 1]) / prev_asset
                    br = (benchmark_close[t] - benchmark_close[t - 1]) / prev_bench
                    cov += (ar - mean_asset) * (br - mean_bench)
                    diff_b = br - mean_bench
                    var_bench += diff_b * diff_b
                cov /= ret_count
                var_bench /= ret_count
                if var_bench > 1e-12:
                    beta = cov / var_bench
                    if beta < -3.0:
                        beta = -3.0
                    elif beta > 3.0:
                        beta = 3.0
            bz = (asset_mom - beta * bench_mom) / peer_std
            if bz < -6.0:
                bz = -6.0
            elif bz > 6.0:
                bz = 6.0
            raw_beta_adjusted[i] = bz

        for i in range(lag, n):
            j = i - lag
            relative_strength_z[i] = raw_relative[j]
            breadth_context[i] = raw_breadth[j]
            beta_adjusted_momentum[i] = raw_beta_adjusted[j]
        return relative_strength_z, breadth_context, beta_adjusted_momentum
else:
    _relative_strength_state_kernel = None


def compute_relative_strength_state(
    close: Sequence[float],
    benchmark_close: Optional[Sequence[float]] = None,
    peer_close_matrix: Optional[Sequence[Sequence[float]]] = None,
    period: int = 63,
    lag: int = 1,
    use_numba: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute lagged cross-sectional relative-strength state."""
    if period < 2:
        raise ValueError("relative-strength period must be at least 2")
    if lag < 1:
        raise ValueError("relative-strength model state must be lagged by at least one bar")
    close_arr = _as_float_array(close, "close")
    if not bool(np.all(np.isfinite(close_arr))):
        raise ValueError("close array must be finite")
    if benchmark_close is None:
        benchmark_arr = close_arr.copy()
    else:
        benchmark_arr = _as_float_array(benchmark_close, "benchmark_close")
        if len(benchmark_arr) != len(close_arr):
            raise ValueError("benchmark_close length must match close")
        if not bool(np.all(np.isfinite(benchmark_arr))):
            raise ValueError("benchmark_close array must be finite")
    peer_matrix = _as_peer_matrix(peer_close_matrix, len(close_arr))
    if use_numba and _relative_strength_state_kernel is not None:
        values = _relative_strength_state_kernel(close_arr, benchmark_arr, peer_matrix, int(period), int(lag))
    else:
        values = _relative_strength_state_reference(close_arr, benchmark_arr, peer_matrix, int(period), int(lag))
    outputs = get_indicator_feature_spec("relative_strength_state").output_names
    return {name: np.ascontiguousarray(arr, dtype=np.float64) for name, arr in zip(outputs, values)}


def build_relative_strength_bundle(
    close: Sequence[float],
    benchmark_close: Optional[Sequence[float]] = None,
    peer_close_matrix: Optional[Sequence[Sequence[float]]] = None,
    period: int = 63,
    lag: int = 1,
    use_numba: bool = True,
) -> IndicatorStateBundle:
    """Build a validated cross-sectional relative-strength state bundle."""
    features = compute_relative_strength_state(
        close,
        benchmark_close=benchmark_close,
        peer_close_matrix=peer_close_matrix,
        period=period,
        lag=lag,
        use_numba=use_numba,
    )
    n_obs = len(next(iter(features.values()))) if features else 0
    bundle = IndicatorStateBundle(
        n_obs=n_obs,
        features=features,
        availability={"relative_strength_state": True},
        spec_names=("relative_strength_state",),
        source_columns=("close",),
    )
    validate_indicator_state_bundle(bundle)
    return bundle


def normalize_indicator_feature_transport(
    feature_matrix: Sequence[Sequence[float]],
    beta: Optional[Sequence[float]] = None,
    sector_scale: Optional[Sequence[float]] = None,
    rolling_window: int = 63,
    clip: float = 6.0,
) -> np.ndarray:
    """Rolling z-normalize indicator features with beta/sector scaling."""
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    if clip <= 0.0:
        raise ValueError("clip must be positive")
    x = np.asarray(feature_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    n, k = x.shape
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if beta is None:
        beta_arr = np.ones(n, dtype=np.float64)
    else:
        beta_arr = np.asarray(beta, dtype=np.float64)
        if beta_arr.ndim == 0:
            beta_arr = np.full(n, float(beta_arr), dtype=np.float64)
        if beta_arr.shape != (n,):
            raise ValueError("beta must be scalar or length n_obs")
        beta_arr = np.nan_to_num(beta_arr, nan=1.0, posinf=1.0, neginf=1.0)
    if sector_scale is None:
        sector_arr = np.ones(n, dtype=np.float64)
    else:
        sector_arr = np.asarray(sector_scale, dtype=np.float64)
        if sector_arr.ndim == 0:
            sector_arr = np.full(n, float(sector_arr), dtype=np.float64)
        if sector_arr.shape != (n,):
            raise ValueError("sector_scale must be scalar or length n_obs")
        sector_arr = np.nan_to_num(sector_arr, nan=1.0, posinf=1.0, neginf=1.0)
    out = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        start = max(0, i - rolling_window + 1)
        denom = np.sqrt(1.0 + 0.25 * beta_arr[i] * beta_arr[i]) * max(abs(sector_arr[i]), 1e-6)
        for j in range(k):
            window = x[start : i + 1, j]
            scale = max(float(np.std(window)), 1e-6)
            z = (x[i, j] - float(np.mean(window))) / scale
            out[i, j] = float(np.clip(z / denom, -clip, clip))
    return np.ascontiguousarray(out, dtype=np.float64)
