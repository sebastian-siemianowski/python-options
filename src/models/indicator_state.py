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
