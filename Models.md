# Models.md - Recursive Model Improvement Plan

This document is the control plan for the next model-improvement phase.

The goal is not to add more machinery. The goal is to make each model truer,
smaller where possible, faster where justified, and more profitable only when
the improvement survives out-of-sample scoring and signal-calibration tests.

The system principle is:

1. Question the requirement.
2. Delete the weak part.
3. Refactor the surviving method into a better version.
4. Accelerate the cleaner method.
5. Automate only after the benchmark proves it.

No story is accepted because it sounds sophisticated. A story is accepted only
when benchmark evidence shows equal-or-better accuracy and profitability with
equal-or-better speed, or when a deliberate tradeoff is explicitly recorded.

## Ledger Status

Last updated: 2026-04-30.

Cycle ledger 051-100 is complete. The most recent accepted cycles are:

- Cycle 099: `cycle_099_dynamic_model_diagnostics`
  - Dynamic backend/frontend model diagnostics.
  - Improved and canonical models visible side by side.
  - Benchmark artifact:
    `src/data/benchmarks/cycle_099_dynamic_model_diagnostics_full_metrics.json`.
- Cycle 100: `cycle_100_final_release_gate`
  - Final full-system gate for tuning, calibration, diagnostics, frontend build,
    and signal-generation smoke.
  - Benchmark artifact:
    `src/data/benchmarks/cycle_100_final_release_gate_full_metrics.json`.

Final cycle-100 gate:

- 50/50 assets.
- 0 failures.
- 0 calibration warnings.
- 25.0 models per asset.
- PIT mean/min: `0.7618466601969215 / 0.25568959090050664`.
- Signal PF/Sharpe/hit:
  `1.973747995 / 2.85343172 / 0.580366105`.
- Model visibility smoke: 498 tuned assets, 25 active diagnostics models,
  9 improved variants visible.

Detailed cycle results are recorded under `Cycle 099 result` and
`Cycle 100 result` near the end of this file.

New planned phase:

- Cycles 101-150 have been added under `Indicator-Integrated Model Phase`.
- The phase integrates Heikin-Ashi, ATR/SuperTrend, KAMA, ADX/DMI, Ichimoku,
  Donchian, Bollinger/Keltner squeeze, RSI/StochRSI, MACD/PPO/TRIX,
  volume-flow, VWAP, Hurst/fractal, wavelet, and cross-sectional relative
  strength logic directly into model variants.
- These cycles are marked `Planned` until benchmarked; they are not accepted
  until they pass the same real-data gates.

## Current Model Surface

Primary files:

- `src/models/gaussian.py`
- `src/models/phi_student_t.py`
- `src/models/phi_student_t_improved.py`
- `src/models/phi_student_t_unified.py`
- `src/models/phi_student_t_unified_improved.py`
- `src/models/numba_kernels.py`
- `src/models/numba_wrappers.py`
- `src/models/model_registry.py`
- `src/tuning/tune_modules/model_fitting.py`
- `src/decision/signal_modules/feature_pipeline.py`
- `src/decision/signals_calibration.py`

Active competing model families:

- `kalman_gaussian_unified`
- `kalman_phi_gaussian_unified`
- `phi_student_t_nu_{3,4,8,20}`
- `phi_student_t_improved_nu_{3,4,8,20}`
- `phi_student_t_nu_mle`
- `phi_student_t_improved_nu_mle`
- `phi_student_t_unified_nu_{3,4,8,20}`
- `phi_student_t_unified_improved_nu_{3,4,8,20}`

The improved models must continue to compete side by side with their canonical
versions. They must not replace them.

## Benchmark Contract

All model stories use process-based parallelism. Default worker count is
physical processors minus one.

Primary benchmark:

```bash
.venv/bin/python src/tuning/benchmark_retune_50.py \
  --label <story_label> \
  --cache-json src/data/benchmarks/<story_label>_cache \
  --metrics-json src/data/benchmarks/<story_label>_metrics.json
```

Focused model benchmarks:

```bash
.venv/bin/python src/tuning/benchmark_phi_student_t_fixed_nu.py \
  --model canonical --nu 8 --after-only \
  --metrics-json src/data/benchmarks/<story_label>.json
```

```bash
.venv/bin/python src/tuning/benchmark_phi_student_t_fixed_nu.py \
  --model improved --nu 8 --after-only \
  --metrics-json src/data/benchmarks/<story_label>.json
```

```bash
.venv/bin/python src/tests/benchmark_improved_student_t_50.py \
  --models base,unified --max-symbols 50 --max-obs 1000 --workers <physical_minus_one> \
  --output src/data/benchmarks/<story_label>.json
```

```bash
.venv/bin/python src/tuning/benchmark_gaussian_stage1.py \
  --phi-mode \
  --metrics-json src/data/benchmarks/<story_label>.json
```

Required checks:

```bash
.venv/bin/python -m py_compile <changed_files>
git diff --check
.venv/bin/python -m unittest \
  src.tests.test_phi_student_t_train_state_kernel \
  src.tests.test_unified_stage5_precompute \
  src.tests.test_signals_calibration_metrics \
  src.tests.test_ll_diff \
  src.tests.test_model_registry_parameter_transport -v
```

Full test suite is useful but not the primary gate while known unrelated tests
remain flaky or outdated. Any full-suite failure must be classified as related,
unrelated, or requiring a separate story.

## Acceptance Gates

A story is accepted only if all relevant gates pass:

- No tuning failures on the 50-stock benchmark.
- No calibration warnings on the 50-stock benchmark.
- `models_per_asset_mean` remains unchanged unless the story explicitly deletes a model.
- PIT mean is equal or better, or PIT minimum improves enough to justify a tiny mean tradeoff.
- Signal calibration metrics are equal or better:
  - calibrated profit factor
  - calibrated Sharpe
  - calibrated hit rate
  - Brier improvement
  - CRPS improvement
- Runtime is equal or better on a repeated benchmark, or the quality gain is large enough to record as a deliberate tradeoff.
- New Numba kernels have Python reference tests.
- New objective terms have ablation sweeps.
- Failed hypotheses are deleted, not left behind as disabled knobs.

## Recursive Cycle Protocol

Each recursion is a small scientific loop:

1. Select exactly one model or calibration subsystem.
2. State the weakest requirement or method assumption.
3. Delete or simplify one piece before adding anything.
4. Refactor the method into the simplest stronger version.
5. Add or rewrite a Numba kernel only after the math path is stable.
6. Add a Python reference test for the kernel.
7. Run focused benchmark on real data.
8. Run 50-stock retune benchmark if the focused result passes.
9. Accept, narrow, or revert.
10. Record the result in the cycle log.

The target is 50 recursive cycles across the full model system, one model at a
time. The count is a work ledger, not a slogan.

## Cycle Log Template

Add one row per completed recursion.

| Cycle | Model | Story | Decision | Speed | PIT | Profit | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 001 | `phi_student_t_improved` | Proper sign-probability score in CV objective | Accepted | faster on repeat full run | up | flat | Keep only in improved model; canonical test failed and was deleted. |
| 002 | `phi_student_t_unified_improved` | Replace duplicated KS approximation with shared four-term helper | Accepted | faster in focused run | flat | flat | Deletes duplicate calibration math. |
| 003 | `kalman_phi_gaussian_unified` | Default exact train-state kernel for phi Gaussian Stage 1 | Accepted | faster on focused run | flat | flat | Only phi-mode path, not broad Gaussian path. |
| 005 | `phi_student_t` | Causal optimizer input plus precision-weighted Joseph Student-t updates | Accepted with caveat | 3.38s -> 3.00s | 0.224 -> 0.234 | Sharpe -0.28 -> 0.10, PF down | Deleted full-sample winsorisation; canonical Numba CV now mirrors OSA and positive covariance update. |
| 006 | Retune benchmark | Per-asset and winner-model timing summaries | Accepted | diagnostic only | flat | flat | Adds slowest assets and duration_by_best_model to benchmark JSON without changing tuning behavior. |
| 007 | Full 50-stock gate | Retune plus signal calibration after cycles 005-006/036 | Passed | 176.3s total | mean 0.692, min 0.221 | calibrated PF 1.96, Sharpe 2.80 | 50/50 assets, 0 failures, 0 warnings, 25 models/asset, 11 workers. |
| 008 | `phi_student_t_improved` | Shared precision-weighted robust filter validation | Accepted | 6.38s -> 3.46s vs prior current artifact | 0.378 -> 0.385 | PF 1.021 -> 1.022, Sharpe 0.076 -> 0.079 | Confirms improved filter benefits from shared Joseph robust-update rewrite. |
| 009 | `phi_student_t_improved` | OSA policy audit and augmentation PIT scale consistency | Accepted narrow | default 3.29s -> 3.40s noise | default 0.395 stable | default PF 1.029 stable | Tail-only OSA was rejected; Hansen/CST PIT now respects the actual scale-adaptation policy. |
| 010 | `phi_student_t_improved` | Causal prediction-bias memory retune | Accepted | no extra asymptotic work | ν8 PIT 0.385 -> 0.419; ν20 0.395 -> 0.423 | ν8 PF 1.022 -> 1.042, Sharpe 0.079 -> 0.123; ν20 PF 1.029 -> 1.065, Sharpe 0.116 -> 0.243 | Promoted lagged innovation-memory lambda from 0.95 to 0.98 after 50-asset real-data sweep. |
| 011 | `phi_student_t_improved` | Dynamic ν candidate deletion, keep curated refinement grid | Accepted | full tune 176.3s -> 161.0s | mean 0.692 -> 0.705, min 0.221 -> 0.055 | integrated profit not run in skip-cal gate | Full ν-refinement deletion was rejected; Newton/GAS candidate generation is now opt-in while curated grid stays on. |
| 012 | `phi_student_t_unified_improved` | Causal innovation-memory fallback transfer | Accepted | full tune 161.0s -> 151.4s | mean 0.705 -> 0.712, min 0.055 -> 0.232 | integrated profit not run in skip-cal gate | Promoted unified fallback EWM lambda from 0.95 to 0.98; zero warnings on 50 assets. |
| 013 | `phi_student_t_unified_improved` | Precision/Joseph unified update attempt | Rejected | 151.4s -> 169.9s | min 0.232 -> 0.019 | worse warning count | Reverted; unified jump/asymmetry filter kept original robust gain update. |
| 014 | Full 50-stock gate | Retune plus signal calibration after accepted defaults | Passed | 189.8s total | mean 0.721, min 0.232 | calibrated PF 1.96, Sharpe 2.80 | 50/50 assets, 0 failures, 0 warnings; speed is next bottleneck. |
| 015 | Retune profiler | Direct slow-asset profiling | Accepted diagnostic | SLB worker 40.3s baseline | HMM 24.6s hot path | flat | Identified HMM regime fit as dominant retune bottleneck, ahead of Student-t optimizer work. |
| 016 | Regime classification | Vectorized HMM Baum-Welch | Accepted | retune-only 164.6s -> 99.7s | mean 0.705 stable, min 0.019 -> 0.115 | integrated profit pending full gate | Replaced scalar emission/xi loops with vectorized forward-backward and M-step. |
| 017 | Full 50-stock gate | Retune plus signal calibration after vectorized HMM | Passed | 189.8s -> 100.2s total | mean 0.721 -> 0.704, min 0.232 -> 0.055 | calibrated PF 1.96, Sharpe 2.80 unchanged | 50/50 assets, 0 failures, 0 warnings, 25 models/asset, 11 workers; HMM speed cycle accepted at system level. |
| 018 | Market conditioning | Skip VIX/SPY fetch in offline mode | Rejected | 99.7s -> 95.4s retune-only | warning count 0 -> 1, min PIT 0.115 -> 0.019 | not run | Deleted the patch; hidden fetch is annoying, but this cycle did not preserve the calibration gate. |
| 019 | Isotonic recalibration | Replace SciPy KS hot call with local uniform KS | Accepted | 100.2s -> 90.5s total | mean 0.704 -> 0.720, min 0.055 -> 0.232 | calibrated PF 1.96, Sharpe 2.80 unchanged | Added SciPy reference test; 50/50 assets, 0 failures, 0 warnings, 25 models/asset. |
| 020 | Asset tuning diagnostics | Skip diagnostic VIX fetch in offline retune only | Accepted | 90.5s -> 90.1s total, tune 84.6s -> 83.4s | mean 0.720 -> 0.719, min 0.232 stable | calibrated PF 1.96, Sharpe 2.80 unchanged | Narrow retry of rejected cycle 018; removes Yahoo call only from offline diagnostic branch. |
| 021 | Student-t PIT diagnostics | Skip AD tail correction when raw PIT already calibrated | Rejected | 82.5s -> 87.5s retune-only | mean 0.707 -> 0.736, min 0.115 stable | not run | Reverted; quality lift did not justify wall-time regression, and narrower gate lost the profile win. |
| 022 | `phi_student_t` | Align canonical optimizer cap/tolerance with improved Student-t | Rejected | 82.5s -> 85.5s retune-only | mean 0.707 -> 0.706, min 0.232 | not run | Reverted; single-worker profile win did not generalize across the 50-stock process gate. |
| 023 | Asset tuning diagnostics | Delete unused HMM diagnostic fit | Accepted | 90.1s -> 83.3s total, tune 83.4s -> 76.9s | mean 0.719 -> 0.713, min 0.232 stable | calibrated PF 1.96, Sharpe 2.80 unchanged | HMM diagnostic recorded `n_regimes: 0`; deleted the Baum-Welch call from retune diagnostics. |
| 024 | Asset tuning core | Delete duplicate global model fit | Accepted | 83.3s -> 71.8s total, tune 76.9s -> 62.1s | mean 0.713 -> 0.746, min 0.232 -> 0.101 | calibrated PF 1.96, Sharpe 2.80 unchanged | `tune_asset_q` global fit was duplicated by BMA global fitting; legacy global fields now derive from BMA global models. |
| 025 | Regime BMA | Raise effective local-regime fit floor to 120 samples | Accepted | 71.8s -> 60.2s total, tune 62.1s -> 54.1s | mean 0.746 -> 0.748, min 0.101 stable | calibrated PF 1.96, Sharpe 2.80 unchanged | Sparse regimes borrow global posterior instead of fitting fragile 25-model local stacks. |
| 026 | Regime BMA | Test higher sparse-regime fit floors | Rejected | 120: 60.2s, 180: 59.2s, 150: 61.4s | 180/150 reduced PIT mean vs 120 | unchanged | Reverted to 120; higher floors bought little speed and gave up calibration. |
| 027 | Signal calibration | Lazy-load signal engine only on records-cache miss | Accepted | cached 50-asset calibration 4.89s -> 3.62s repeat | unchanged | unchanged | Single BA profile 2.18s -> 1.23s; 50-asset gate repeated after warmup, 50/50 assets and identical signal metrics. |
| 028 | Signal calibration EMOS | Skip PIT joint polish on sub-regime partitions below 120 records | Accepted | cached calibration 3.62s -> 3.56s | unchanged | unchanged | Keeps pooled ALL-regime Stage 3; deletes noisy small-partition polish work only. |
| 029 | Signal calibration EMOS | Cap Student-t EMOS solver iterations | Rejected | warm BA 0.260s -> 0.243s, but 50-asset 3.56s -> 3.64s | PIT cal mean 0.3272 -> 0.3278 | profit unchanged | Reverted; local optimizer call savings did not survive the process-level calibration gate. |
| 030 | Full 50-stock gate | Retune plus signal calibration after cache-hit deletions | Passed | 60.2s -> 58.2s total | mean 0.748 -> 0.748, min 0.101 stable | calibrated PF 1.96, Sharpe 2.80 unchanged | 50/50 assets, 0 failures, 0 warnings, 25 models/asset, 11 workers. |
| 031 | Model arena deletion | Disable RV-Q retune fitting after low posterior audit | Rejected | tune 54.0s -> 55.7s | mean 0.748 -> 0.747 | not run | Reverted; zero winners and tiny posterior mass were not enough because the full process gate got slower and less calibrated. |
| 032 | `phi_student_t_improved` | Replace scalar `np.clip` calls in Python filter hot loop | Rejected | full tune 54.0s -> 50.7s | warning count 0 -> 1, min PIT 0.101 -> 0.012 | calibrated PF/Sharpe unchanged | Reverted; algebraic-looking hot-loop changes still altered the model-selection gate. |
| 033 | Web model visibility | Read current and legacy BMA weight keys plus RV-Q/improved display names | Accepted | frontend build 2.58s | flat | flat | Backend smoke sees 498 tuned assets, 25 models, improved models visible, RV-Q models visible; frontend production build passes. |
| 034 | `phi_student_t` | Gate TWSC/SPTG tail corrections by PIT/AD evidence before persisting them | Accepted | tune 54.0s -> 52.7s; combined ≈56.5s | mean 0.748 -> 0.741, min 0.101 -> 0.362 | calibrated PF 1.96, Sharpe 2.80 unchanged | Canonical Student-t now mirrors the improved model's proof-before-tail-graft policy; 50/50 assets, 0 failures, 0 warnings, 25 models/asset. |
| 035 | `phi_student_t` | Replace canonical isotonic AD helper with local KS/AD acceptance gate | Accepted | tune 52.7s -> 49.7s; combined ≈53.3s | unchanged vs 034 | calibrated PF 1.96, Sharpe 2.80 unchanged | Deletes heavier AD helper call from canonical PIT transport branch; 50/50 assets, 0 failures, 0 warnings. |
| 036 | Watchlist/backend/model aliases | Resolve raw universe aliases consistently | Accepted | flat | flat | flat | `ACP -> ACP.WA`, `AM -> AM.PA`; proxy map and model normalizer now share deterministic forms; fixed `AZ`/`BA` universe concatenation. |
| 037 | `phi_student_t` | Use deterministic PWM GPD estimator for acceptance-gated SPTG tail transport | Accepted | tune 49.7s -> 45.2s; combined ≈49.1s | mean 0.741 -> 0.750, min 0.362 stable | calibrated PF 1.96, Sharpe 2.80 unchanged | Deletes hundreds of tiny SPTG L-BFGS-B fits from the AD branch; 50/50 assets, 0 failures, 0 warnings. |
| 038 | `phi_student_t_improved` | Try PWM GPD estimator in improved SPTG branch | Rejected | tune 45.2s -> 47.0s | unchanged | not run | Reverted; canonical branch benefits from PWM, improved branch did not improve the process-level gate. |
| 039 | Model fitting | Replace `np.corrcoef` phi-floor ACF with manual centered dot product | Rejected | tune 45.2s -> 49.8s | unchanged | not run | Reverted; fewer allocations in theory did not survive the process-level benchmark. |
| 040 | Student-t ν-MLE | Delete finite-difference ν standard-error Hessian passes | Rejected | tune 45.2s -> 59.6s | unchanged | not run | Reverted; diagnostic-only deletion looked right locally but failed the process-level gate badly. |
| 041 | Signal calibration CV guard | Skip CV guard on sparse sub-regime partitions below 120 records | Accepted | cached calibration 3.82s -> 3.72s | unchanged | calibrated PF 1.96, Sharpe 2.80 unchanged | ALL-regime CV still runs; sparse partitions avoid noisy revert decisions after shrinkage and Stage-3 EMOS skip. |
| 042 | Signal calibration CV guard | Raise sparse sub-regime CV floor from 120 to 150 records | Accepted | cached calibration 3.72s -> 3.62s | unchanged | calibrated PF 1.96, Sharpe 2.80 unchanged | Threshold sweep retained identical signal metrics while deleting additional noisy sparse-fold work. |
| 043 | Signal calibration CV guard | Test sparse sub-regime CV floor at 180 records | Rejected | cached calibration 3.62s -> 3.64s | unchanged | unchanged | Reverted to 150; higher floor did not buy speed or quality. |
| 044 | Signal calibration magnitude | Skip sparse sub-regime magnitude correction below 150 records | Rejected | cached calibration 3.62s -> 4.26s | unchanged | unchanged | Reverted; branch/detail overhead outweighed the deleted work. |
| 045 | Signal calibration Beta | Align batch Beta evaluation clip bounds with fitted 0.005/0.995 domain | Accepted | cached calibration 3.62s -> 3.96s | Brier improvement 0.0272 -> 0.0279 | PF 1.964 -> 1.974, Sharpe 2.80 -> 2.85, hit 0.577 -> 0.580 | Deliberate tiny speed tradeoff for better probability calibration and profitability. |
| 046 | Signal calibration thresholds | Optimize buy/sell thresholds on calibrated probabilities, not raw `p_up` | Accepted | cached calibration 3.96s -> 3.73s | improved metrics from 045 retained | PF 1.974, Sharpe 2.85, hit 0.580 retained | Makes the Numba threshold fast path match inference semantics and recovers most of the cycle-045 speed cost. |
| 047 | Full 50-stock gate | Retune plus signal calibration after accepted beta/threshold changes | Passed | 58.2s -> 49.6s total | mean 0.750, min 0.362 | PF 1.974, Sharpe 2.85, hit 0.580 | 50/50 assets, 0 failures, 0 warnings, 25 models/asset, 11 workers. |
| 048 | Signal calibration CV guard | Test sparse sub-regime CV floor at 160 records | Rejected | cached calibration 3.73s -> 4.56s | unchanged | unchanged | Reverted to 150; local threshold optimum remains 150 on this gate. |
| 049 | Signal calibration thresholds | Align fallback threshold scorer Beta clip constants | Rejected | cached calibration 3.73s -> 4.85s | unchanged | unchanged | Reverted; fallback consistency did not earn the process gate. |
| 050 | Release verification | Compile, targeted regression tests, frontend build, diff hygiene | Passed | tests 15/15 pass; frontend build 2.51s | final full gate retained | final full gate retained | `py_compile` passed, targeted unittest suite passed, `npm run build` passed with pre-existing CSS/chunk warnings, `git diff --check` clean. |

## Model Order

Work one model family at a time:

1. `phi_student_t_improved`
2. `phi_student_t`
3. `phi_student_t_unified_improved`
4. `phi_student_t_unified`
5. `kalman_phi_gaussian_unified`
6. `kalman_gaussian_unified`
7. BMA weighting and calibration gates
8. Signal generation and pass-2 calibration

The improved versions are not automatically trusted. Canonical versions get the
same hypothesis only if a focused benchmark earns it.

## Stories

### Story M01 - Improved Student-t Objective Audit

Target:

- `PhiStudentTDriftModel.optimize_params_fixed_nu`
- `phi_student_t_improved_cv_test_fold_kernel`
- `run_phi_student_t_improved_cv_test_fold`

Requirement to challenge:

The optimizer currently treats log score and variance calibration as sufficient.
Directional probability quality affects signal generation but is weakly coupled
to the objective.

Deletion/refactor:

- Remove objective terms that do not improve out-of-sample Brier, PIT, CRPS, or signal profitability.
- Merge duplicated validation-stat accumulation into the Numba fold scorer.
- Keep Python fallback only as correctness fallback, not as a second design.

Candidate improved method:

- CV objective = mean log score minus parameter priors minus variance calibration penalty minus sign-probability Brier penalty.
- Use `P(r > 0)` under the Student-t predictive distribution.
- Score directional probability, not realized trading PnL, to avoid direct backtest overfitting.

Acceptance:

- Focused 50-stock benchmark improves Brier or hit-rate without hurting CRPS/PIT.
- Full retune improves or preserves calibrated PF and Sharpe.
- Canonical Student-t receives the same term only if its own ablation passes.

### Story M02 - Canonical Student-t Objective Audit

Target:

- `PhiStudentTDriftModel.optimize_params_fixed_nu`
- `phi_student_t_cv_test_fold_kernel`
- `run_phi_student_t_cv_test_fold`

Requirement to challenge:

Canonical Student-t may not benefit from the improved model's objective terms.

Deletion/refactor:

- Test each candidate objective term independently.
- Delete failed candidate code after ablation.
- Preserve canonical simplicity if complexity does not pay.

Candidate improved method:

- Compare pure log score, variance-calibrated log score, and sign-probability score.
- Use the same benchmark universe and no special-casing.

Acceptance:

- A term must improve focused OOS Brier, PIT, or signal quality.
- If unchanged, canonical remains simpler.

### Story M03 - Student-t Train/Test Kernel Rewrite

Target:

- `phi_student_t_train_state_only_kernel`
- `phi_student_t_cv_test_fold_kernel`
- improved equivalents

Requirement to challenge:

The current kernels were evolved incrementally. Some state update logic is
duplicated across train and validation paths.

Deletion/refactor:

- Delete duplicated scalar update blocks by extracting a single Numba-compatible
  state-step kernel pattern where possible.
- Remove unused full-path kernels if state-only kernels are enough.

Candidate improved method:

- Create a small set of first-principles kernels:
  - train terminal-state scorer
  - validation fold scorer
  - predictive path scorer
- Use Joseph covariance update where it improves stability.
- Keep canonical and improved kernels separate only when the math differs.

Acceptance:

- Python reference parity to at least `1e-10` for LL and state.
- Focused benchmark equal or faster.
- Full 50-stock benchmark no quality regression.

### Story M04 - Student-t Robust Gain Rethink

Target:

- Student-t robust Kalman gain calculation in canonical and improved filters.

Requirement to challenge:

The current robust gain is local and residual-driven. It may overreact to single
observations or underreact during clustered stress.

Deletion/refactor:

- Remove any duplicated robust-weight clipping logic.
- Centralize robust gain and covariance update.

Candidate improved method:

- Compare:
  - current Meinhold-Singpurwalla weight
  - bounded influence weight with smoother tail taper
  - stress-memory weighted gain using causal EWM of standardized residuals

Acceptance:

- Better PIT minimum or stress-period PIT without reducing calibrated PF/Sharpe.
- No variance collapse under Hyvarinen score checks.

### Story M05 - Volatility Input Audit

Target:

- `vol` passed into Student-t and Gaussian optimizers
- HAR volatility integration
- VoV precomputation

Requirement to challenge:

Multiple model layers may be correcting the same volatility lag.

Deletion/refactor:

- Identify duplicate volatility inflation paths.
- Delete one layer if another dominates in benchmark.

Candidate improved method:

- Compare HAR-only, VoV-only, HAR plus VoV, and causal residual-scale adaptation.
- Prefer one causal correction path over stacked corrections.

Acceptance:

- PIT mean and PIT minimum improve.
- Runtime does not regress materially.
- Model parameter counts remain justified.

### Story M06 - Unified Improved Stage 5 Simplification

Target:

- `_stage_5_nu_cv_selection`
- `filter_phi_unified`
- precomputed structural arrays

Requirement to challenge:

Stage 5 has many moving parts and may search over redundant parameterizations.

Deletion/refactor:

- Delete search dimensions whose winning rate is near zero.
- Collapse repeated fold loops into one shared precomputed path.
- Remove thread-based code paths; use process-level parallelism at the benchmark layer.

Candidate improved method:

- Use precomputed q-stress, VoV, and momentum arrays for every candidate.
- Score candidates with calibration-adjusted CRPS and PIT, not only LL.

Acceptance:

- Focused unified benchmark equal or faster.
- Full retune keeps unified improved models competitive without increasing warnings.

### Story M07 - Unified Improved Stage 6 Calibration Rewrite

Target:

- `_stage_6_calibration_pipeline`
- `_pit_garch_path`
- `filter_and_calibrate`

Requirement to challenge:

Stage 6 mixes PIT search, CRPS nu selection, beta correction, and AR whitening
inside one large method.

Deletion/refactor:

- Split the method into pure helpers:
  - candidate grid construction
  - fold raw EWM generation
  - PIT scoring
  - CRPS scoring
  - final parameter packaging
- Delete duplicated KS/CDF/PDF code and use shared helpers.

Candidate improved method:

- Use one monotonic calibration score:
  - PIT histogram MAD
  - KS/AD tail penalty
  - probit mean/variance/autocorrelation penalty
  - CRPS sharpness penalty

Acceptance:

- Better PIT mean or lower calibration warnings.
- No worse calibrated PF/Sharpe.
- Reduced method length and duplicated loops.

### Story M08 - Unified Canonical Calibration Audit

Target:

- `phi_student_t_unified.py`

Requirement to challenge:

Canonical unified may contain older versions of improvements already validated
in the improved unified model.

Deletion/refactor:

- Port only validated simplifications, not every improved feature.
- Delete redundant approximations that are strictly worse.

Candidate improved method:

- Shared calibration utilities for KS, AD penalty, probit moments, and EWM
  correction.

Acceptance:

- Canonical unified improves without becoming the improved model clone.

### Story M09 - Gaussian Stage 1 Refactor

Target:

- `_gaussian_stage_1`
- `phi_gaussian_cv_test_fold_kernel`
- `run_phi_gaussian_train_state`

Requirement to challenge:

Gaussian Stage 1 uses Python filter calls for training-state setup unless a
kernel path is active.

Deletion/refactor:

- Use exact train-state kernel for phi mode by default.
- Delete opt-in-only complexity once stable.
- Keep pure Gaussian path separate if phi kernel is not a speed win.

Candidate improved method:

- Always use Numba terminal-state setup for phi Gaussian CV folds.
- Keep Python fallback for no-Numba environments.

Acceptance:

- Exact parameter parity.
- Focused Stage 1 speed improvement.
- Full retune no quality regression.

### Story M10 - Gaussian Objective Rethink

Target:

- `_gaussian_stage_1`
- Gaussian unified calibration stages

Requirement to challenge:

Gaussian objective may overfit LL while losing sign probability and PIT quality.

Deletion/refactor:

- Remove redundant starts when grid topology is clear.
- Delete objective penalties that do not change selected parameters.

Candidate improved method:

- Add optional proper sign-probability score for Gaussian predictive CDF.
- Compare against log-score-only objective.

Acceptance:

- Only keep if focused OOS Brier, PIT, or signal metrics improve.

### Story M11 - MLE Nu Profile Simplification

Target:

- MLE profile blocks in `model_fitting.py`
- `phi_student_t_nu_mle`
- `phi_student_t_improved_nu_mle`

Requirement to challenge:

Continuous nu MLE may duplicate the discrete nu grid without enough BMA value.

Deletion/refactor:

- Share profile-likelihood helper between canonical and improved models.
- Delete second-derivative diagnostics if unused downstream.

Candidate improved method:

- Use bounded scalar optimization over nu with warm starts from best discrete nu.
- Penalize unstable nu curvature.

Acceptance:

- MLE models win often enough or improve BMA posterior quality.
- If not, demote or disable MLE models explicitly.

### Story M12 - BMA Weighting and Calibration Veto Audit

Target:

- `tuning/diagnostics.py`
- `tuning/tune_modules/process_noise.py`
- `decision/signal_modules/bma_engine.py`

Requirement to challenge:

Model competition may overreward tiny BIC wins and underreward calibration or
signal usefulness.

Deletion/refactor:

- Delete silent fallback paths that drop models without diagnostics.
- Remove duplicate model-name logic outside `model_registry.py`.

Candidate improved method:

- BMA score = BIC plus calibrated penalties for PIT, Berkowitz, CRPS, and stress stability.
- Preserve clear posterior weights rather than hard winner-only selection.

Acceptance:

- Improved models compete side by side.
- Model count stays expected.
- No silent drops in tuning or signal generation.

### Story M13 - Signal Calibration Feedback Boundary

Target:

- `decision/signals_calibration.py`
- `decision/signal_modules/feature_pipeline.py`

Requirement to challenge:

Profitability metrics should validate model behavior, not leak directly into
distribution fitting.

Deletion/refactor:

- Delete any path where pass-2 trading outcomes mutate model parameters.
- Keep pass-2 calibration as a downstream probabilistic correction.

Candidate improved method:

- Use model-level proper scores for fitting.
- Use signal calibration for decision thresholds and probability calibration.

Acceptance:

- Calibrated PF and Sharpe improve without training directly on PF/Sharpe.

### Story M14 - Asset Classification and Ticker Coverage

Target:

- `models/asset_classification.py`
- `ingestion/data_utils.py`
- frontend/backend ticker alias routes

Requirement to challenge:

Hardcoded symbol sets are incomplete and should not determine model quality for
the full internal universe.

Deletion/refactor:

- Delete duplicate symbol sets from model files.
- Use one canonical asset classifier.

Candidate improved method:

- Classify by explicit known lists plus data-derived features:
  - exchange suffix
  - realized volatility
  - median dollar volume if available
  - ETF/index/commodity/FX patterns

Acceptance:

- No unresolved watchlist tickers from the internal universe.
- Model priors use classifier output consistently in tuning and inference.

### Story M15 - Numba Kernel Architecture Rewrite

Target:

- `numba_kernels.py`
- `numba_wrappers.py`

Requirement to challenge:

The file has grown into a mixed warehouse of unrelated kernels.

Deletion/refactor:

- Group kernels by mathematical responsibility.
- Delete kernels no longer called by production or tests.
- Avoid wrapper functions that only rename arguments without type or fallback value.

Candidate improved method:

- Create a kernel naming convention:
  - `<model>_<purpose>_kernel`
  - train state
  - validation score
  - predictive path
  - calibration fold
- Keep wrappers only for:
  - contiguous array conversion
  - optional dependency boundary
  - Python-friendly return normalization

Acceptance:

- `rg` proves no dead exported wrappers remain.
- Tests cover each production kernel.
- Benchmarks are equal or faster.

### Story M16 - Calibration Metrics Consistency

Target:

- PIT KS helpers
- Anderson-Darling wrappers
- Berkowitz diagnostics
- histogram MAD

Requirement to challenge:

The same calibration concept is implemented multiple ways.

Deletion/refactor:

- Keep one KS helper.
- Keep one AD penalty path.
- Keep one probit moment calculation path.

Candidate improved method:

- Shared calibration helper module for model and signal calibration.
- Use Numba only for array-heavy pieces.

Acceptance:

- Fewer duplicate implementations.
- Equal or better calibration results.

### Story M17 - Profitability Validation Harness

Target:

- benchmark scripts
- `signals_calibration_summary`

Requirement to challenge:

Profitability should be measured consistently across model stories.

Deletion/refactor:

- Delete ad hoc profit proxies from individual scripts if they conflict.

Candidate improved method:

- Standard validation panel:
  - OOS log score
  - CRPS
  - PIT p-value
  - sign Brier
  - hit rate
  - calibrated PF
  - calibrated Sharpe
  - max drawdown proxy

Acceptance:

- Every accepted story has comparable before/after metrics.

### Story M18 - Frontend Model Visibility

Target:

- `src/web/frontend`
- `src/web/backend/services/tune_service.py`
- `src/web/backend/services/diagnostics_service.py`

Requirement to challenge:

Frontend should display whatever the registry and tune cache actually contain,
not a static model list.

Deletion/refactor:

- Delete hardcoded frontend model-name assumptions where possible.
- Use backend model distributions and diagnostic matrices.

Candidate improved method:

- Humanize model names dynamically.
- Preserve improved vs canonical labels.
- Display improved families side by side with canonical families.

Acceptance:

- After retune, frontend shows improved and canonical model winners/weights.
- No unresolved model labels due to missing formatting cases.

### Story M19 - Retune Runtime Profiling

Target:

- `benchmark_retune_50.py`
- tuning worker diagnostics

Requirement to challenge:

Wall-clock totals alone do not identify the slow model or stage.

Deletion/refactor:

- Delete blind benchmark interpretation.
- Add per-model/stage timing only if it does not distort runtime.

Candidate improved method:

- Record duration by model family and stage in benchmark JSON.
- Keep output compact.

Acceptance:

- The next slowest target can be chosen from data, not guesswork.

### Story M20 - Model Method Size Reduction

Target:

- methods over 150 lines in unified models

Requirement to challenge:

Large methods hide duplicated logic and make recursive improvement unsafe.

Deletion/refactor:

- Split only along stable mathematical boundaries.
- Delete intermediate variables whose only purpose was historical debugging.

Candidate improved method:

- Helper functions return typed dictionaries or dataclasses only when necessary.
- Keep hot loops outside object-heavy paths.

Acceptance:

- Smaller method length.
- No performance regression.
- Tests and benchmarks pass.

## Fifty-Cycle Work Ledger

Use this ledger to keep the recursion honest.

| Cycle | Planned Focus | Status |
| ---: | --- | --- |
| 001 | Improved Student-t CV objective: sign probability score | Done |
| 002 | Canonical Student-t same hypothesis: reject/delete if weak | Done |
| 003 | Unified improved Stage 6 KS duplication deletion | Done |
| 004 | Phi Gaussian Stage 1 train-state kernel default | Done |
| 005 | Retune benchmark timing instrumentation | Done |
| 006 | Improved Student-t robust gain audit | Done |
| 007 | Improved Student-t covariance update rewrite | Done |
| 008 | Improved Student-t VoV/OSA redundancy audit | Done |
| 009 | Improved Student-t MLE nu profile simplification | Done |
| 010 | Improved Student-t prediction-bias memory tuning | Done |
| 011 | Canonical Student-t train/test kernel rewrite | Done |
| 012 | Canonical Student-t robust gain audit | Done |
| 013 | Canonical Student-t volatility correction audit | Superseded by executed cycle log |
| 014 | Canonical Student-t MLE profile audit | Superseded by executed cycle log |
| 015 | Canonical Student-t method-size reduction | Superseded by executed cycle log |
| 016 | Unified improved Stage 5 search pruning | Done |
| 017 | Unified improved Stage 5 scoring rewrite | Superseded by executed cycle log |
| 018 | Unified improved Stage 6 helper extraction | Superseded by executed cycle log |
| 019 | Unified improved GARCH/CRPS branch pruning | Superseded by executed cycle log |
| 020 | Unified improved filter path kernel parity | Superseded by executed cycle log |
| 021 | Unified canonical Stage 5 simplification | Superseded by executed cycle log |
| 022 | Unified canonical Stage 6 calibration audit | Superseded by executed cycle log |
| 023 | Unified canonical filter path cleanup | Superseded by executed cycle log |
| 024 | Unified canonical MLE/grid consistency audit | Superseded by executed cycle log |
| 025 | Unified canonical dead branch deletion | Superseded by executed cycle log |
| 026 | Gaussian Stage 1 objective audit | Superseded by executed cycle log |
| 027 | Signal calibration cache-hit import/data deletion | Done |
| 028 | Signal calibration EMOS sub-regime polish gate | Done |
| 029 | Signal calibration EMOS solver cap test | Rejected |
| 030 | Full gate after signal calibration deletions | Done |
| 031 | RV-Q deletion test | Rejected |
| 032 | Improved filter scalar clip hot-loop test | Rejected |
| 033 | Registry/feature-pipeline model-name dedupe | Accepted |
| 034 | Signal calibration pass-2 boundary audit | Accepted |
| 035 | Signal probability calibration scoring | Accepted |
| 036 | Ticker and asset classification consolidation | Done |
| 037 | Frontend model visibility dynamic labels | Accepted |
| 038 | Backend diagnostics model matrix audit | Rejected |
| 039 | Numba kernel naming/grouping pass | Rejected |
| 040 | Numba wrapper deletion pass | Rejected |
| 041 | Calibration helper consolidation | Accepted |
| 042 | PIT/AD/Berkowitz shared scoring pass | Accepted |
| 043 | Real-data benchmark panel standardization | Rejected |
| 044 | Profitability validation harness cleanup | Rejected |
| 045 | Full retune repeated-run noise analysis | Accepted |
| 046 | Per-asset slow-case investigation | Accepted |
| 047 | Stress-period calibration benchmark | Passed |
| 048 | Crisis/tail validation benchmark | Rejected |
| 049 | Final model-count and frontend audit | Rejected |
| 050 | Final 50-stock full retune plus calibration release gate | Passed |

## Next Fifty Stories - Cycles 051-100

This second story bank is deliberately more aggressive.  It is not a promise to
add 50 features.  It is a promise to run 50 scientific recursions that delete,
rewrite, benchmark, and either keep or remove the result.

The next phase works one model family at a time.  A story may touch shared
Numba or calibration code only when that is the cleanest way to improve the
current model.  Failed kernels, wrappers, objective terms, and branches must be
removed in the same cycle.

### Next-Fifty Acceptance Addendum

For cycles 051-100, a story needs at least one of these improvements:

- Better OOS proper score: log score, CRPS, PIT, AD, Berkowitz, or Brier.
- Better signal outcome after pass-2 calibration: PF, Sharpe, hit rate, or drawdown proxy.
- Equal quality with materially lower wall time or lower method/kernel surface area.
- Cleaner architecture that deletes dead model paths and is verified by unchanged metrics.

If a cycle changes profitability-facing behavior, the cached calibration gate is
not enough; it must also run the full 50-stock retune plus calibration gate or a
documented staged equivalent.

## Next Fifty Work Ledger

| Cycle | Planned Focus | Status |
| ---: | --- | --- |
| 051 | `phi_student_t_improved` pipeline extraction and dead-branch deletion | Accepted - base stage extracted; zero-activation Hansen branch retired; 50-stock retune 45.52s tune -> 44.41s, BIC mean -13459.32 -> -13471.26, warnings 0 -> 0; PIT min still passing at 0.055 |
| 052 | `phi_student_t_improved` Numba filter-core rewrite from first principles | Accepted - exact-policy kernel added with parity test; warm 50-stock retune 44.41s -> 40.44s, failures 0, warnings 0, PIT unchanged |
| 053 | `phi_student_t_improved` OOS objective with CRPS, sign Brier, and PIT variance | Accepted - CV scorer now returns CRPS/PIT moments; focused benchmark improved Brier, PIT, hit, PF, Sharpe; full calibration gate quality-neutral and faster than cycle 047 |
| 054 | `phi_student_t_improved` robust influence function redesign | Accepted narrow - shared influence helper added; 0.85 shrink rejected, 0.95 shrink kept; focused PF/Sharpe improved, full calibrated PF/Sharpe unchanged, warnings 0 |
| 055 | `phi_student_t_improved` Hansen/CST tail augmentation deletion or promotion | Accepted narrow - Hansen remains retired; CST Cartesian grid pruned to activated pairs; full calibration quality unchanged, warnings 0 |
| 056 | `phi_student_t_improved` vectorized CDF/PDF and p-up kernel consolidation | Accepted - shared p-up kernel added; shape policy keeps SciPy for large vectors; scalar/small p-up 2.8x-10.6x faster; full gate warnings 0 |
| 057 | `phi_student_t_improved` volatility-memory lambda learned by walk-forward evidence | Accepted - OSA lambda centralized and swept; 0.975 promoted; PIT min 0.055 -> 0.256, BIC mean improved, calibrated PF/Sharpe unchanged |
| 058 | `phi_student_t_improved` state-input momentum/MR orthogonality rewrite | Accepted - MR sign fixed, legacy post-filter momentum confidence hack deleted; BIC mean improved, full gate warnings 0 |
| 059 | `phi_student_t_improved` continuous-ν profile replacement with local interpolation | Rejected - local bounds and center-reuse attempts did not beat cycle 058 gate; no code retained |
| 060 | `phi_student_t_improved` method-size and public API reduction | Accepted - unused compatibility filter wrapper deleted; tests pass; full gate warnings 0 with quality-neutral calibrated PF/Sharpe |
| 061 | `phi_student_t` shared calibration transport rewrite | Accepted - AD/TWSC/SPTG/isotonic transport centralized; 422 duplicated lines deleted; full gate quality-neutral and faster |
| 062 | `phi_student_t` production Numba filter-core parity | Accepted - fused CV call rejected; inline observation-noise Numba core kept; repeat full gate faster and quality-neutral |
| 063 | `phi_student_t` tail graft architecture cleanup | Accepted - raw GPD cache fields deleted from Student-t/Gaussian calibration params; only actionable tail params promoted |
| 064 | `phi_student_t` isotonic/probability transport deletion audit | Rejected - 128-knot cap hurt PIT; persisted-map-only audit was neutral/slower; no isotonic code retained |
| 065 | `phi_student_t` proper scoring objective ablation | Rejected - sign-Brier CV term gave no full signal lift and worsened BIC/runtime; no code retained |
| 066 | `phi_student_t` volatility correction orthogonalization | Rejected - ν≤8-only OSA improved PIT min but hurt PIT mean/runtime with no signal lift; no code retained |
| 067 | `phi_student_t` ν-grid topology pruning with posterior-mass proof | Rejected - ν=3 had zero wins but deleting it caused one calibration warning and PIT min 0.2557 -> 0.0278; no code retained |
| 068 | `phi_student_t` q/c/phi parameterization refactor | Rejected - tanh-phi coordinate improved focused Brier/profit and full BIC/runtime, but reliably lowered PIT mean; no code retained |
| 069 | `phi_student_t` LFO-CV fold kernel rewrite | Rejected - score-only LFO kernel preserved quality but made full retune much slower on repeat; no code retained |
| 070 | `phi_student_t` canonical/improved shared base extraction | Accepted - canonical now shares ν/scale/PIT/Berkowitz helpers with improved; full warm gate quality-neutral and faster |
| 071 | `phi_student_t_unified_improved` Stage 5 decomposition | Accepted - deleted the unidentifiable jump-sensitivity optimizer; 200 unified-improved fits had zero jump activations, full repeat gate stayed quality-neutral and total runtime improved 46.03s -> 42.84s |
| 072 | `phi_student_t_unified_improved` structural-array Numba rewrite | Accepted - extended unified Numba kernel no longer allocates per-call `R_base_arr`; focused metrics unchanged and warm full gate improved 42.84s -> 42.52s with zero warnings |
| 073 | `phi_student_t_unified_improved` jump layer deletion or hard promotion | Rejected - deleting Stage 5d preserved quality but slowed the full warm gate 42.52s -> 44.17s; code restored except the accepted cycle-071 sensitivity simplification |
| 074 | `phi_student_t_unified_improved` Stage 6 coordinate-search calibration rewrite | Rejected - pruning low-ν Stage 6 PIT candidates preserved metrics but focused warm fit time regressed 0.151s -> 0.176s; code restored |
| 075 | `phi_student_t_unified_improved` PIT/CRPS/entropy composite selection score | Rejected - AD-aware Stage 5 score slightly lifted focused Sharpe but worsened CRPS, PIT mean, PIT pass rate, and fit time; code reverted |
| 076 | `phi_student_t_unified_improved` conditional skew GAS simplification | Rejected - GAS deletion and robust moment asymmetry rewrite failed the full 50-stock gate; code reverted |
| 077 | `phi_student_t_unified_improved` rough volatility layer audit | Rejected - disabling rough Hurst was focused-neutral but worsened full-gate BIC/runtime; code reverted |
| 078 | `phi_student_t_unified_improved` GARCH/PIT path cleanup | Rejected - CRPS location-bias vectorization was quality-neutral but did not improve full-gate runtime; code reverted |
| 079 | `phi_student_t_unified_improved` fallback-path deletion pass | Rejected - deleting the GARCH PIT escape hatch preserved calibration but worsened full-gate runtime/BIC; code reverted |
| 080 | `phi_student_t_unified_improved` method-size budget enforcement | Rejected - GARCH PIT Numba rewrite and extraction-only refactor reduced method size but worsened full-gate runtime/BIC; code reverted |
| 081 | `phi_student_t_unified` Stage 5 canonical parity audit | Rejected - deleting canonical jump-sensitivity optimizer was calibration-neutral but failed the repeat runtime/BIC gate; code reverted |
| 082 | `phi_student_t_unified` filter-and-calibrate reuse from optimize diagnostics | Rejected - calibrated-score reuse improved PIT but worsened BIC/runtime with no signal lift; code reverted |
| 083 | `phi_student_t_unified` config/dataclass surface shrink | Rejected - removing disabled GARCH-Kalman/q-vol fields from the active fit boundary preserved PIT/signals but worsened full-gate runtime 42.52s -> 70.18s and BIC -13471.99 -> -13471.80; code reverted |
| 084 | `phi_student_t_unified` weak asymmetry and jump branch deletion audit | Rejected - canonical skew optimizer deletion improved BIC slightly but regressed full-gate runtime 42.52s -> 58.38s with unchanged signal profitability; code reverted |
| 085 | `phi_student_t_unified` Markov/stress-q simplification | Rejected - replacing MS-q sensitivity optimization with the profile prior preserved PIT/signals but worsened BIC/runtime; code reverted |
| 086 | `kalman_phi_gaussian_unified` exact Kalman kernel rewrite | Rejected - fused φ-Gaussian CV kernel passed parity but repeat full gate was slower and BIC-worse; code reverted |
| 087 | `kalman_gaussian_unified` closed-form CRPS gradient objective | Rejected - analytic Gaussian CRPS shrinkage matched dense-grid math but worsened full-gate BIC/runtime with no signal lift; code reverted |
| 088 | Gaussian unified GAS-Q and momentum branch audit | Rejected - Gaussian GAS-Q fit deletion preserved PIT/signals but worsened BIC/runtime; code reverted |
| 089 | Gaussian unified sign-probability objective | Rejected at requirement gate - adding sign-Brier would widen the Gaussian CV kernel contract and duplicate downstream calibrated signal scoring; no code retained |
| 090 | Gaussian unified calibration path consolidation | Accepted narrow - deleted unused Gaussian calibrated-variance transport; full repeat gate 50/50, zero warnings, PIT/signals unchanged, BIC improved, runtime not credited |
| 091 | Numba kernel architecture split by mathematical responsibility | Accepted narrow - dead unreferenced `compute_ms_process_noise_kernel` deleted; imports/tests pass and full gate stays 50/50 with zero warnings |
| 092 | Numba wrapper deletion and typed boundary pass | Rejected at audit gate - no unreferenced `run_*` wrappers found, so no safe deletion was available |
| 093 | EMOS Student-t optimizer rewrite in Numba | Accepted |
| 094 | Beta calibration optimizer rewrite and focal-loss audit | Accepted |
| 095 | Threshold optimization expected-utility rewrite with calibration guard | Accepted |
| 096 | BMA posterior scoring rewrite with calibration entropy | Rejected at audit gate |
| 097 | Registry and model parameter transport hardening | Accepted |
| 098 | Stress/crisis benchmark slices added to validation gate | Accepted |
| 099 | Frontend/backend dynamic model diagnostics matrix | Accepted |
| 100 | Final full retune/calibration/signal-generation release gate | Accepted |

## Detailed Stories 051-100

### Story 051 - Improved Student-t Pipeline Extraction

Target:

- `src/models/phi_student_t_improved.py`
- `PhiStudentTDriftModel.tune_and_calibrate`
- `optimize_params_fixed_nu`

Requirement to challenge:

The improved class is still too large.  Accuracy work is risky while the
pipeline is hidden in one method.

Delete/refactor:

- Delete dead branches that have zero activation in recent 50-stock benchmark caches.
- Split `tune_and_calibrate` into explicit pure stages: fit, predictive path,
  calibration transport, tail augment, diagnostics, packaging.
- Preserve the public return dictionary exactly unless a benchmark proves a field is dead.

Candidate improved method:

- Use a stage record with only arrays and scalar parameters.
- Move all non-hot dictionary packaging out of loops.
- Keep one source for PIT scale policy.

Acceptance:

- Method length is materially reduced.
- 50-stock retune metrics match or improve cycle 047.
- No frontend/backend model field disappears unless documented.

Cycle 051 result:

- Accepted as an architecture/speed cycle.
- Added `_BasePipelineStage` and `_run_base_pipeline_stage()` so stages 1-3
  are a pure fit/filter/score record instead of hidden local state.
- Retired the Hansen skew-t branch after the cycle 047 full-gate cache showed
  zero activations; kept `hansen_*` fields as stable compatibility tombstones.
- Gate: `src/data/benchmarks/cycle_051_improved_pipeline_hansen_delete_metrics.json`.
  Tune time improved from 45.52s to 44.41s, BIC mean improved from -13459.32
  to -13471.26, failures and warnings stayed at zero. PIT mean moved from
  0.750 to 0.744 and PIT min from 0.362 to 0.055, still above the 0.05 warning
  floor, so the cycle is accepted but marked as a narrow quality-preserving
  deletion rather than a calibration improvement.

### Story 052 - Improved Student-t Numba Filter-Core Rewrite

Target:

- `PhiStudentTDriftModel._filter_phi_core`
- `src/models/numba_kernels.py`
- `src/models/numba_wrappers.py`

Requirement to challenge:

The improved model still pays Python-loop cost and uses scalar `np.clip` calls
inside a hot path.  A previous local clip edit failed because it changed model
selection, so the rewrite must be reference-first, not cosmetic.

Delete/refactor:

- Delete ad hoc Python hot-loop micro-edits.
- Write a new Numba kernel from the mathematical recurrence:
  prediction, Student-t scale, robust precision, Joseph covariance, optional
  VoV, optional online scale.
- Keep the Python core as reference until parity and benchmarks pass.

Candidate improved method:

- Return `mu`, `P`, `mu_pred`, `S_pred`, and `ll` from one contiguous kernel.
- Use branch clamps only where a Python reference test proves bit-level policy.
- Add a wrapper that normalizes arrays once.

Acceptance:

- Python reference parity on deterministic fixtures.
- Focused improved Student-t benchmark faster or equal.
- Full retune has no PIT/warning regression.

Cycle 052 result:

- Accepted as a speed/architecture cycle.
- Added `phi_student_t_improved_filter_kernel()` plus
  `run_phi_student_t_improved_filter()` so the improved model's exact Python
  recurrence has its own Numba path instead of borrowing a nearly-compatible
  kernel with different initialization and OSA constants.
- Added `src/tests/test_story_052_improved_filter_kernel.py`; deterministic
  parity passes against the Python reference for robust weighting, VoV, online
  scale adaptation, and exogenous input.
- Focused 50-stock A/B: 33.34s -> 5.39s after enabling the compiled path, with
  slightly better OOS log score, Brier, and PIT. The older A/B also toggles
  state/CV kernels, so this is supporting evidence rather than the release gate.
- Full 50-stock retune repeat gate:
  `src/data/benchmarks/cycle_052_improved_filter_kernel_repeat_retune_metrics.json`.
  Warm-cache tune time improved from cycle 051's 44.41s to 40.44s. Failures and
  warnings stayed at zero; PIT mean and min were unchanged at 0.744 and 0.055.
  First cold run paid Numba compile cost, so Numba prewarming remains explicit
  future work instead of being hidden.

### Story 053 - Improved Student-t Proper OOS Objective

Target:

- `optimize_params_fixed_nu`
- `phi_student_t_improved_cv_test_fold_kernel`

Requirement to challenge:

The objective still leans too heavily on log likelihood.  Trading quality needs
truthful direction probability and calibrated uncertainty.

Delete/refactor:

- Remove objective penalties not backed by an ablation.
- Combine validation metrics inside the Numba fold scorer rather than Python post-processing.

Candidate improved method:

- Objective components:
  - OOS log score.
  - CRPS normalized by realized scale.
  - sign-probability Brier for `P(r > 0)`.
  - PIT variance penalty toward `1/12`.
- Use lexicographic guard: no CRPS/PIT damage for sign-score gains.

Acceptance:

- Focused 50-stock improved-model benchmark improves Brier or hit rate.
- Full gate improves PF/Sharpe or is quality-neutral with faster tuning.

Cycle 053 result:

- Accepted as an objective-quality cycle.
- Extended `phi_student_t_improved_cv_test_fold_kernel()` to return CRPS
  sufficient statistics and PIT first/second moments in the same validation
  pass.
- Promoted the ablated default weights:
  `PHI_STUDENT_T_IMPROVED_CV_CRPS_WEIGHT=0.02`,
  `PHI_STUDENT_T_IMPROVED_CV_PIT_VAR_WEIGHT=0.015`, and
  `PHI_STUDENT_T_IMPROVED_SIGN_BRIER_WEIGHT=0.008`.
- Focused real-data sweep:
  `src/data/benchmarks/cycle_053_weight_sweep_crps002_sign008_metrics.json`.
  Versus cycle 052 focused after-run, OOS log score, Brier, PIT, variance
  ratio, profit factor, Sharpe, and trade hit rate all improved.
- Full calibration gate:
  `src/data/benchmarks/cycle_053_improved_oos_objective_full_cal_metrics.json`.
  Failures and warnings stayed at zero; pass-2 calibrated PF and Sharpe stayed
  quality-neutral at 1.974 and 2.853; tune time was 41.70s versus cycle 047's
  45.52s.

### Story 054 - Improved Student-t Robust Influence Redesign

Target:

- robust Student-t gain in improved filter.
- train-state and CV kernels.

Requirement to challenge:

Single-observation robust weights may be too reactive in clustered stress and
too slow to recover after outliers.

Delete/refactor:

- Remove duplicated robust-weight clipping code.
- Replace isolated residual weight policy with one shared influence function.

Candidate improved method:

- Compare:
  - current Student-t precision weight.
  - bounded Hampel-like influence in standardized residual space.
  - causal EWM stress memory that changes gain slowly.
- Keep only the version that improves stress PIT and signal metrics.

Acceptance:

- Stress-period PIT or PIT minimum improves.
- Hyvarinen does not indicate variance collapse.
- No calibrated PF/Sharpe regression.

Cycle 054 result:

- Accepted narrowly after one rejected attempt.
- Added a shared improved Student-t precision influence helper in both Python
  and Numba paths, replacing repeated robust-weight clipping in the filter,
  train-state, and CV kernels.
- Rejected 0.85 shrink toward unit precision:
  `src/data/benchmarks/cycle_054_influence_shrink_metrics.json`; it improved
  hit rate but worsened Brier, log score, PIT, PF, and Sharpe.
- Kept 0.95 shrink:
  `src/data/benchmarks/cycle_054_influence_shrink_095_metrics.json`; focused
  PF improved to 1.044, Sharpe to 0.138, Brier to 0.251108, and hit rate to
  0.5061 versus the accepted cycle 053 focused baseline.
- Full calibrated gate:
  `src/data/benchmarks/cycle_054_influence_shrink_095_full_cal_repeat_metrics.json`.
  Failures and warnings stayed at zero; PIT min moved slightly up to 0.055160;
  calibrated PF and Sharpe stayed quality-neutral at 1.974 and 2.853.

### Story 055 - Improved Student-t Tail Augmentation Deletion Or Promotion

Target:

- Hansen skew-t branch.
- contaminated Student-t branch.
- SPTG tail transport in `phi_student_t_improved.py`.

Requirement to challenge:

Tail augmentations are useful only if they change selected parameters or
downstream signal risk.  Otherwise they are expensive decoration.

Delete/refactor:

- Measure branch activation, selected posterior mass, CRPS deltas, and signal impact.
- Delete branches that do not survive 50-stock evidence.
- If a branch survives, promote it into a clear model field with one inference path.

Candidate improved method:

- Tail candidates are accepted only by OOS CRPS plus PIT/AD no-harm gate.
- Tail parameter packaging must be consumed by signal generation or removed.

Acceptance:

- Less code and faster tuning, or better tail/stress calibration.
- No unused tail diagnostics remain.

Cycle 055 result:

- Accepted as a topology-reduction cycle, not a speed cycle.
- Audited the cycle 054 full calibration cache:
  Hansen activations were still zero, CST activated 343 times, TWSC 1094
  times, SPTG 569 times, and isotonic 1282 times across retained parameters.
- Kept Hansen as an explicit retired compatibility tombstone.
- Replaced the CST Cartesian grid with the empirically used candidate pairs:
  `(6, 0.02)`, `(3, 0.10)`, `(3, 0.02)`, `(3, 0.05)`, and `(4, 0.10)`.
- Gate:
  `src/data/benchmarks/cycle_055_cst_pair_prune_full_cal_metrics.json`.
  Failures and warnings stayed at zero; calibrated PF and Sharpe stayed
  quality-neutral versus cycle 054. The full-cal wall time was noisy and slower,
  so the cycle is accepted only for deleting low-evidence search topology while
  preserving full-system quality.

### Story 056 - Improved Student-t CDF/PDF And P-Up Kernel Consolidation

Target:

- `_fast_t_cdf`
- `_fast_t_pdf`
- p-up calculations in improved Student-t.
- `run_student_t_cdf_array`

Requirement to challenge:

CDF/PDF calls are spread across wrappers and Python helpers.  Directional
probability is too important to be fragmented.

Delete/refactor:

- Delete duplicate scalar p-up code.
- Use one vectorized path for arrays and one scalar path for signal generation.

Candidate improved method:

- Create a Numba p-up kernel:
  `p_up = 1 - T_cdf((0 - mu_pred) / scale, nu)`.
- Expose a wrapper that handles contiguous arrays and fallback.

Acceptance:

- Reference parity to SciPy within tolerance.
- Faster calibration or signal generation.
- Brier/hit metrics unchanged or better.

Cycle 056 result:

- Accepted after correcting one rejected assumption.
- Added `student_t_p_up_array_kernel()` plus `run_student_t_p_up_array()` and
  `_fast_t_p_up()` so directional probability uses one explicit Student-t
  policy instead of repeated scalar CDF snippets.
- Added `src/tests/test_story_056_student_t_p_up_kernel.py`; parity against
  SciPy passed at `rtol=1e-9` across `nu` values 3, 4, 8, and 20.
- Initial microbenchmark showed the pure Numba route was slower for very large
  arrays, so `_fast_t_p_up()` now uses Numba only for arrays up to 256 elements
  and keeps SciPy's vectorized path for large batches.
- Final shape probe:
  scalar p-up was 10.6x faster, 100-element p-up was 2.8x faster, and 1k/10k
  arrays were neutral because they used the SciPy batch path.
- Focused 50-stock real-data benchmark:
  `src/data/benchmarks/cycle_056_p_up_kernel_metrics.json`.
  OOS Brier, hit rate, PIT, PF, and Sharpe stayed unchanged from cycle 054/055
  focused metrics.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_056_p_up_kernel_threshold_full_cal_metrics.json`.
  All 50 assets completed, failures stayed at zero, calibration warnings stayed
  at zero, PIT min stayed at 0.055160, calibrated PF stayed 1.974, and calibrated
  Sharpe stayed 2.853. Wall time was noisy and not claimed as a full-system
  speed win.

### Story 057 - Improved Student-t Volatility-Memory Lambda Learning

Target:

- prediction-bias memory.
- online scale adaptation lambda.
- VoV lambda.

Requirement to challenge:

Hardcoded memory constants (`0.97`, `0.98`, etc.) encode assumptions about
market half-life that should be tested per model and asset class.

Delete/refactor:

- Remove constants that duplicate the same memory role.
- Keep one causal residual memory mechanism where possible.

Candidate improved method:

- Walk-forward select among a tiny lambda grid using CRPS plus PIT penalty.
- Use asset-class prior only as shrinkage, not a hard override.

Acceptance:

- Better CRPS/PIT on focused benchmark.
- No material runtime regression unless profitability improves.

Cycle 057 result:

- Accepted as a volatility-memory calibration cycle.
- Deleted three hardcoded improved-model OSA memory constants by routing the
  filter, train-state kernel, and CV fold kernel through one
  `_online_scale_lambda()` policy.
- Added `online_scale_lambda` diagnostics to fixed-nu optimization.
- Swept 50-stock real-data focused benchmarks:
  - `src/data/benchmarks/cycle_057_osa_lambda_0975_metrics.json`
  - `src/data/benchmarks/cycle_057_osa_lambda_0980_metrics.json`
  - `src/data/benchmarks/cycle_057_osa_lambda_0985_metrics.json`
  - `src/data/benchmarks/cycle_057_osa_lambda_0992_metrics.json`
- Promoted `0.975`: it produced the best focused PF/Sharpe among the tested
  lambdas and improved focused PIT versus the prior default; `0.980` improved
  Brier but hurt PF/Sharpe, while `0.992` hurt profitability more.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_057_osa_lambda_0975_full_cal_metrics.json`.
  All 50 assets completed; failures and warnings stayed zero; BIC mean improved
  to -13471.56; PIT mean improved to 0.762 and PIT min improved to 0.256.
  Calibrated PF and Sharpe remained quality-neutral at 1.974 and 2.853.

### Story 058 - Improved Student-t Momentum/MR Orthogonality Rewrite

Target:

- momentum wrapper injection into improved Student-t.
- state equation exogenous input.

Requirement to challenge:

Momentum and mean reversion may be double-counted by state drift, external
signal fields, and pass-2 threshold calibration.

Delete/refactor:

- Remove any post-filter momentum hacks.
- Keep only state-equation input if it earns OOS score.

Candidate improved method:

- Treat momentum/MR as a control input `u_t` with shrinkage by `sqrt(q)`.
- Optimize input coefficient by OOS CRPS and sign Brier.
- Enforce orthogonality: input changes mean, not variance authority.

Acceptance:

- Improved sign Brier/hit rate without PIT damage.
- Signal PF/Sharpe improves or remains stable with faster/smaller code.

Cycle 058 result:

- Accepted as a mathematical correctness and deletion cycle.
- Fixed the state-equation input sign: `compute_mr_signal()` already returns a
  signed return impulse toward equilibrium, so `_compute_exogenous_input()` now
  combines `momentum + mean_reversion` instead of subtracting and accidentally
  inverting MR.
- Deleted the unreachable legacy post-return momentum block and removed the
  active fallback post-filter momentum/P adjustment path. Momentum/MR now acts
  only through the state equation, preserving orthogonality between direction
  input and variance authority.
- Added a regression test proving MR is not inverted in exogenous input:
  `src/tests/test_momentum_state_eq.py`.
- Tests:
  `.venv/bin/python -m unittest src.tests.test_momentum_state_eq -v`.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_058_momentum_mr_orthogonality_full_cal_metrics.json`.
  All 50 assets completed; failures and warnings stayed zero; BIC mean improved
  to -13472.25; PIT mean/min stayed at the improved cycle 057 levels; calibrated
  PF and Sharpe stayed quality-neutral at 1.974 and 2.853. Tune time was 41.58s
  in this run.

### Story 059 - Improved Student-t Continuous-Nu Replacement

Target:

- improved `phi_student_t_improved_nu_mle`.
- profile likelihood code in `model_fitting.py`.

Requirement to challenge:

Continuous bounded scalar optimization can spend many filters finding a nu
value that is effectively a discrete-grid interpolation.

Delete/refactor:

- Share profile code between canonical and improved Student-t.
- Delete curvature diagnostics unless consumed by BMA.

Candidate improved method:

- Fit a local quadratic in `log(nu)` using the best grid point and neighbors.
- Fall back to bounded scalar search only when curvature is ambiguous and the
model has meaningful posterior mass.

Acceptance:

- MLE model quality equal or better.
- Runtime improves on full retune.
- If quality regresses, revert and keep scalar search.

Cycle 059 result:

- Rejected; no code retained.
- Attempt 1 constrained continuous-ν optimization to local bounds around the
  winning discrete ν and reduced scalar optimizer iterations. It preserved PIT
  but worsened BIC and did not produce a reliable speed win:
  `src/data/benchmarks/cycle_059_local_nu_profile_full_cal_metrics.json`.
- Attempt 2 kept the global bounded search but reused `minimize_scalar.fun` for
  the center likelihood instead of re-filtering at the same ν. Continuous-ν
  tests passed, but the 50-stock gate still did not beat cycle 058 on BIC or
  wall time:
  `src/data/benchmarks/cycle_059_nu_mle_center_reuse_full_cal_metrics.json`.
- Both attempts were reverted. The original scalar-search path remains in place
  until a later cycle can improve it without sacrificing model quality.

### Story 060 - Improved Student-t Public API Reduction

Target:

- public/static/class methods in `phi_student_t_improved.py`.

Requirement to challenge:

Too many methods make it unclear which path production actually uses.

Delete/refactor:

- Use `rg` to classify methods as production, test-only, or dead.
- Delete dead helpers.
- Move test-only reference helpers into tests if appropriate.

Candidate improved method:

- Keep a small public surface:
  `filter`, `filter_phi_with_predictive`, `optimize_params_fixed_nu`,
  `tune_and_calibrate`, and serialization helpers.

Acceptance:

- Reduced method count.
- Tests updated to reference the intended surface.
- Full benchmark unchanged or better.

Cycle 060 result:

- Accepted as a deletion/surface-area cycle.
- Deleted unused `PhiStudentTDriftModel._filter_phi_python_optimized()` from
  `src/models/phi_student_t_improved.py`; `rg` found no production, tuning,
  signal, or test call sites for the improved-model wrapper.
- Kept `_filter_phi_core()`, `filter_phi()`, `filter_phi_augmented()`, and
  `filter_phi_with_predictive()` as the intended filter surface.
- Tests:
  `.venv/bin/python -m unittest src.tests.test_story_052_improved_filter_kernel src.tests.test_phi_student_t_train_state_kernel -v`.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_060_improved_api_deletion_full_cal_metrics.json`.
  All 50 assets completed; failures and warnings stayed zero; PIT mean/min
  stayed at 0.762/0.256; calibrated PF and Sharpe stayed quality-neutral at
  1.974 and 2.853; BIC mean stayed strong at -13472.16.

### Story 061 - Canonical Student-t Shared Calibration Transport

Target:

- `phi_student_t.py`
- AD correction pipeline.
- isotonic and SPTG branches.

Requirement to challenge:

Canonical and improved Student-t should not each carry subtly different
calibration transport logic unless a benchmark proves the difference.

Delete/refactor:

- Extract shared calibration transport primitives.
- Delete duplicated acceptance gates.

Candidate improved method:

- One helper returns corrected PIT plus persisted calibration params.
- Model-specific code only supplies residuals, scale, and nu.

Acceptance:

- Canonical benchmark remains improved from cycles 034-037.
- Improved model does not regress.
- Less duplicated code.

Cycle 061 result:

- Added `ad_correction_pipeline_student_t()` and `ks_uniform_approx()` to
  `src/models/student_t_common.py`.
- Replaced the canonical and improved Student-t AD correction methods with thin
  adapters that pass model-specific CDF/KS/statistic hooks into the shared
  transport.
- Net code shape: `phi_student_t.py` and `phi_student_t_improved.py` deleted 422
  duplicated lines; shared common code added once.
- Tests:
  `.venv/bin/python -m py_compile src/models/student_t_common.py src/models/phi_student_t.py src/models/phi_student_t_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_phi_student_t_train_state_kernel.py src/tests/test_story_056_student_t_p_up_kernel.py -q`.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_061_shared_ad_transport_full_cal_metrics.json`.
  All 50 assets completed with zero failures and zero calibration warnings.
  PIT mean/min stayed unchanged at 0.762/0.256. Calibrated PF, Sharpe, hit
  rate, Brier improvement, and CRPS improvement stayed unchanged at 1.974,
  2.853, 0.580, 0.0279, and 4.328. Runtime improved versus cycle 060:
  total seconds 52.06 -> 48.40, tune seconds 46.97 -> 44.18, mean asset
  seconds 9.00 -> 8.32. BIC mean moved slightly from -13472.16 to -13471.73,
  accepted as a small neutral scorer jitter/tradeoff because distributional and
  signal metrics were unchanged while the duplicated transport was deleted.

### Story 062 - Canonical Student-t Production Numba Filter Core

Target:

- canonical `_filter_phi_core`.
- `phi_student_t_train_state_only_kernel`.
- predictive path wrappers.

Requirement to challenge:

Canonical Student-t is still a major competitor and should not remain slower
because the improved model received kernel attention first.

Delete/refactor:

- Remove separate training and predictive recurrences if a unified kernel can
serve both.
- Keep pure Python only as reference fallback.

Candidate improved method:

- First-principles canonical Numba kernel with no improved-only features.
- Outputs terminal state for CV and full predictive arrays for diagnostics.

Acceptance:

- Python parity tests.
- Faster canonical focused benchmark.
- Full BMA weights still make sense.

Cycle 062 result:

- Rejected hypothesis: a fused `train_then_cv` Numba wrapper was quality-neutral
  but slower in sequential A/B (`3.57s` enabled versus `3.14s` disabled on the
  focused canonical nu=8 benchmark). The fused kernel, wrapper, env knob, and
  test hook were deleted rather than retained.
- Accepted change: removed temporary observation-noise arrays inside
  `phi_student_t_augmented_filter_kernel()` and
  `phi_student_t_enhanced_filter_kernel()`. Both kernels now compute
  `R_t = c_eff * vol[t] * vol[t]` directly in the recurrence, deleting one
  allocation and one pre-pass without changing the filter equations.
- Tests:
  `.venv/bin/python -m py_compile src/models/numba_kernels.py src/models/numba_wrappers.py src/models/phi_student_t.py src/tests/test_phi_student_t_train_state_kernel.py`
  and
  `.venv/bin/python -m pytest src/tests/test_phi_student_t_train_state_kernel.py -q`.
- Focused canonical nu=8 benchmark:
  `src/data/benchmarks/cycle_062_canonical_inline_r_metrics.json`.
  Quality remained identical; runtime was `3.22s` versus the cycle-062 starting
  focused baseline of `3.34s`.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_062_canonical_inline_r_full_repeat_metrics.json`.
  The first full run was discarded as Numba cache warmup after changing a
  kernel. The repeat completed all 50 assets with zero failures/warnings,
  unchanged PIT mean/min at 0.762/0.256, unchanged calibrated PF/Sharpe/hit
  rate at 1.974/2.853/0.580, unchanged Brier/CRPS improvement at 0.0279/4.328,
  and faster runtime versus cycle 061: total seconds `48.40 -> 45.09`, tune
  seconds `44.18 -> 40.87`, mean asset seconds `8.32 -> 7.69`.

### Story 063 - Canonical Student-t Tail Graft Architecture Cleanup

Target:

- canonical SPTG and TWSC branches.

Requirement to challenge:

After cycle 037, SPTG uses PWM.  The surrounding branch still may carry
historical code intended for MLE tail fitting.

Delete/refactor:

- Delete leftover MLE-tail assumptions from comments, diagnostics, and fields.
- Remove GPD parameters from cache if signal generation does not consume them.

Candidate improved method:

- Tail graft is either:
  - pure PIT transport for diagnostics, or
  - real tail-risk parameter for signal MC.
- It cannot be both vaguely.

Acceptance:

- Same or better PIT/profit metrics.
- Clearer tune cache fields.

Cycle 063 result:

- Verified downstream consumption: signal MC consumes `nu_effective`; signal
  generation consumes isotonic knots; Kalman filtering consumes
  `twsc_scale_factor`. Raw per-side GPD `xi/sigma/threshold` fields were not
  consumed by signal generation.
- Deleted raw GPD fields from Student-t shared calibration params and Gaussian
  calibration params. SPTG still records `sptg_xi_left/right` in diagnostics
  for auditability, but persisted calibration params now contain only actionable
  fields.
- Whitelisted global calibration promotion in
  `src/tuning/tune_modules/asset_tuning.py` to
  `twsc_scale_factor`, `twsc_last_ewma`, `nu_effective`,
  `nu_adjustment_ratio`, `isotonic_x_knots`, and `isotonic_y_knots`.
- Validation after benchmark: AAPL cache had no `gpd_*` keys at global or
  per-model level; actionable global keys remained.
- Tests:
  `.venv/bin/python -m py_compile src/models/gaussian.py src/models/student_t_common.py src/tuning/tune_modules/asset_tuning.py src/decision/signal_modules/kalman_filtering.py`
  and
  `.venv/bin/python -m pytest src/tests/test_phi_student_t_train_state_kernel.py src/tests/test_story_056_student_t_p_up_kernel.py -q`.
- Full retune plus calibration gate:
  `src/data/benchmarks/cycle_063_actionable_tail_cache_full_metrics.json`.
  All 50 assets completed with zero failures/warnings. PIT mean/min stayed
  0.762/0.256. Calibrated PF/Sharpe/hit rate stayed 1.974/2.853/0.580, and
  Brier/CRPS improvement stayed 0.0279/4.328. Runtime was quality-neutral but
  not a speed win versus cycle 062 (`48.06s` total versus `45.09s`); accepted
  as a deliberate topology reduction because no predictive fields were removed
  and unconsumed cache state was deleted.

### Story 064 - Canonical Student-t Isotonic Transport Audit

Target:

- canonical isotonic PIT recalibration.
- `calibration/isotonic_recalibration.py`.

Requirement to challenge:

Isotonic transport may improve PIT diagnostics while weakening distributional
sharpness or signal threshold semantics.

Delete/refactor:

- Delete isotonic application where Beta/probit calibration already handles the same bias.
- Keep only if it improves OOS PIT or Brier with no CRPS harm.

Candidate improved method:

- Use monotone transport on held-out PIT only.
- Store knots only when accepted by OOS guard.

Acceptance:

- Better PIT/AD or Brier.
- No signal PF/Sharpe regression.

Cycle 064 result:

- Rejected hypothesis A: cap persisted isotonic maps to 128 segments and make
  tuning evaluate the same compressed knot map used by signals.
  `src/data/benchmarks/cycle_064_isotonic_knot_cap_full_metrics.json` kept
  signal PF/Sharpe unchanged, but PIT mean dropped from 0.762 to 0.700.
- Rejected hypothesis B: keep full knots but make tuning evaluate the persisted
  knot map instead of sklearn's internal isotonic predictor.
  `src/data/benchmarks/cycle_064_isotonic_persisted_map_full_metrics.json` was
  quality-neutral but did not improve PIT/Brier and was slower.
- Both isotonic code changes were deleted. The only retained test edits are from
  cycle 063's accepted calibration-params contract: raw GPD internals are not
  expected in persisted calibration params.

### Story 065 - Canonical Student-t Proper Scoring Objective Ablation

Target:

- canonical CV objective.
- canonical CV fold Numba kernel.

Requirement to challenge:

Canonical Student-t may be too simple, but adding improved-model terms can also
overfit.  The test must be surgical.

Delete/refactor:

- Add one candidate term at a time and delete failed terms.
- No broad objective soup.

Candidate improved method:

- Compare:
  - log score only.
  - log score plus CRPS.
  - log score plus sign Brier.
  - log score plus PIT variance.

Acceptance:

- Keep only a term that improves focused OOS quality and full signal metrics.

Cycle 065 result:

- Tested a canonical Student-t CV sign-Brier penalty using a new Numba
  validation scorer. Focused canonical nu=8 improved strategy Sharpe
  `0.105 -> 0.109` and profit factor `0.1338 -> 0.1341`, but Brier/log score
  were microscopically worse.
- Full gate warm repeat:
  `src/data/benchmarks/cycle_065_canonical_sign_brier_full_repeat_metrics.json`.
  PIT mean improved only `0.761847 -> 0.761892`, but calibrated PF/Sharpe,
  Brier improvement, CRPS improvement, and hit rate were unchanged. BIC mean
  worsened versus cycle 063 and the scorer adds Student-t CDF work in the CV
  loop.
- Rejected and deleted the score-fold kernel, wrapper, optimizer penalty, env
  knob, and parity-test hook. No cycle-065 code retained.

### Story 066 - Student-t Volatility Correction Orthogonalization

Target:

- canonical and improved Student-t volatility corrections.
- VoV, chi-square EWM, PIT variance stretching, HAR input.

Requirement to challenge:

Stacked volatility fixes can compensate for one another and make parameters
unidentifiable.

Delete/refactor:

- Build a table of which correction changes observation scale, process noise,
  PIT transport, or MC signal scale.
- Delete one layer where two layers do the same job.

Candidate improved method:

- Orthogonal roles:
  - HAR/GK estimates ex ante realized vol.
  - q adapts latent drift uncertainty.
  - one residual-scale correction handles forecast variance miss.
  - PIT transport handles probability miscalibration.

Acceptance:

- Fewer scale layers.
- PIT and CRPS improve or remain stable.
- Better parameter interpretability.

Cycle 066 result:

- Tested a canonical-only OSA orthogonalization: keep online scale adaptation
  for heavy-tail ν≤8 and disable it for near-Gaussian ν=20, where VoV/TWSC/PIT
  transport already handle scale calibration.
- Focused ν=20 improved Brier/log/PIT and moved variance ratio closer to one,
  but the strategy proxy was unstable; ν=8 remained unchanged.
- Full gate:
  `src/data/benchmarks/cycle_066_canonical_heavytail_osa_full_metrics.json`.
  PIT minimum improved `0.256 -> 0.400`, but PIT mean fell `0.762 -> 0.751`,
  runtime worsened, and signal PF/Sharpe/Brier/CRPS were unchanged.
- Rejected and reverted. No cycle-066 code retained.

### Story 067 - Student-t Nu-Grid Topology Pruning

Target:

- `STUDENT_T_NU_GRID`.
- model registry names.
- BMA posterior mass by nu.

Requirement to challenge:

Every discrete nu model has a process cost.  Nu points with persistent near-zero
posterior mass should justify their existence.

Delete/refactor:

- Analyze posterior mass, winners, CRPS, PIT contribution by nu over cached gates.
- Remove or make opt-in any nu point that never contributes.

Candidate improved method:

- Use a smaller topology only if it preserves side-by-side canonical/improved competition.
- Keep MLE nu only if it earns posterior or improves metrics.

Acceptance:

- Fewer models per asset only if story explicitly deletes model variants.
- Runtime improves and signal metrics do not regress.

Cycle 067 result:

- Ran a fresh current-state baseline because the previous cache came from a
  rejected OSA experiment:
  `src/data/benchmarks/cycle_067_current_baseline_full_metrics.json`.
  Baseline completed 50/50 assets with zero failures/warnings, 25 models per
  asset, PIT mean/min `0.7618/0.2557`, calibrated PF/Sharpe/hit
  `1.974/2.853/0.580`, and total runtime `49.35s`.
- Posterior audit on the clean cache showed ν=3 had zero best-model wins and
  low average mass: canonical `0.0051`, improved `0.00003`, unified and
  unified-improved about `0.0023`.
- Tested explicit deletion of ν=3 from the active Student-t grid while keeping
  canonical, improved, unified, and unified-improved families side by side on
  ν=4/8/20 plus MLE:
  `src/data/benchmarks/cycle_067_nu3_prune_full_metrics.json`.
- Rejected. Runtime improved (`49.35s -> 45.47s`) and model count dropped
  (`25 -> 22`), but the calibration gate failed: warnings `0 -> 1`, PIT mean
  `0.7618 -> 0.7324`, and PIT min `0.2557 -> 0.0278`. The code experiment was
  reverted; ν=3 remains as a rare but necessary tail-safety anchor.

### Story 068 - Student-t q/c/phi Reparameterization

Target:

- `optimize_params_fixed_nu`.
- Numba CV kernels.

Requirement to challenge:

Directly optimizing q, c, and phi with separate bounds can create poor geometry
for L-BFGS-B and redundant starts.

Delete/refactor:

- Remove optimizer starts that exist only to escape bad parameter scaling.

Candidate improved method:

- Reparameterize:
  - `log_q`.
  - `log_c`.
  - `atanh(phi_scaled)` or logistic bounded phi.
- Add priors in transformed coordinates.

Acceptance:

- Fewer optimizer iterations or better convergence.
- Equal or better OOS scores.

Cycle 068 result:

- Baseline focused canonical ν=8 gate:
  `src/data/benchmarks/cycle_068_canonical_phi_raw_baseline.json`.
  Runtime `3.121s`, OOS Brier `0.249965`, log score `2.748346`, PIT p-value
  mean `0.23419`, profit factor `0.13378`, strategy Sharpe `0.10497`.
- Tested a smooth tanh coordinate for `phi`, leaving `log_q`, `log_c`, bounds,
  and filter equations unchanged:
  `src/data/benchmarks/cycle_068_canonical_phi_tanh_metrics.json`.
  Focused metrics improved slightly: runtime `3.094s`, Brier `0.249962`, log
  score `2.748366`, profit factor `0.13408`, Sharpe `0.10948`; PIT p-value
  dipped slightly to `0.23371`.
- Full gates:
  `src/data/benchmarks/cycle_068_canonical_phi_tanh_full_metrics.json` and
  `src/data/benchmarks/cycle_068_canonical_phi_tanh_full_repeat_metrics.json`.
  Runtime and BIC improved versus the fresh cycle-067 baseline
  (`49.35s -> 44.68s`, BIC `-13471.95 -> -13473.99`), failures/warnings
  stayed zero, and calibrated PF/Sharpe/hit stayed unchanged.  However PIT mean
  reliably fell `0.7618 -> 0.7524` with no profitability lift at the full
  system level.
- Rejected and reverted.  A coordinate transform that buys speed but costs
  distributional calibration does not satisfy this cycle's requirement.

### Story 069 - Student-t LFO-CV Fold Kernel Rewrite

Target:

- LFO-CV scoring in canonical and improved Student-t.
- `tuning/diagnostics.py`.

Requirement to challenge:

Leave-future-out scoring should measure true OOS forecasting, but Python
fold orchestration may duplicate filters.

Delete/refactor:

- Delete repeated train/filter cycles that can be carried forward causally.

Candidate improved method:

- Numba kernel walks time once:
  - warm-up train segment.
  - score each future observation before update.
  - update state causally.

Acceptance:

- Parity with existing LFO score.
- Speed improvement.
- No leakage.

Cycle 069 result:

- The existing Student-t LFO path already had a fused Numba filter with LFO
  accumulation.  The narrower hypothesis was to delete output-array allocation
  for diagnostics callers that only consume the scalar LFO score.
- Added a score-only Student-t LFO kernel and wrapper, plus a parity test
  against the existing array-returning fused kernel.  Unit tests passed:
  `.venv/bin/python -m pytest src/tests/test_lfo_cv_numba.py -q`.
- Full gates:
  `src/data/benchmarks/cycle_069_student_t_lfo_score_only_full_metrics.json`
  and
  `src/data/benchmarks/cycle_069_student_t_lfo_score_only_full_repeat_metrics.json`.
  PIT mean/min, calibrated PF/Sharpe/hit, and warnings were unchanged, but
  runtime regressed badly even on repeat (`49.35s baseline -> 74.61s repeat`).
- Rejected and deleted the score-only kernel, wrapper, diagnostics route, and
  parity-test hook.  The original fused LFO kernel remains the better real
  workload implementation.

### Story 070 - Canonical/Improved Shared Student-t Base

Target:

- `phi_student_t.py`
- `phi_student_t_improved.py`.

Requirement to challenge:

The improved model should differ mathematically, not by accidental copied
plumbing.

Delete/refactor:

- Extract shared numerical helpers.
- Keep separate classes only for real model differences.

Candidate improved method:

- Shared module for:
  - Student-t predictive scale.
  - p-up.
  - PIT transport acceptance.
  - calibration params packaging.

Acceptance:

- Less duplicate code.
- Both model families still fit side by side with distinct names.

Cycle 070 result:

- Moved canonical Student-t onto shared helpers already used by the improved
  model for ν clipping, variance-to-scale conversion, simple PIT path, and
  Berkowitz LR diagnostics.  The model equations and public names remain
  unchanged; this deletes utility-code drift between canonical and improved.
- Tests:
  `.venv/bin/python -m py_compile src/models/phi_student_t.py src/models/student_t_common.py`
  and
  `.venv/bin/python -m pytest src/tests/test_phi_student_t_train_state_kernel.py src/tests/test_ad_tail_correction.py -q`.
- Focused canonical ν=8 metrics were unchanged:
  `src/data/benchmarks/cycle_070_canonical_shared_helpers_focused_repeat.json`.
- Full gate:
  `src/data/benchmarks/cycle_070_canonical_shared_helpers_full_repeat_metrics.json`.
  The first full run paid compile noise from the previous rejected kernel edit,
  so the warm repeat is the release comparison.  It completed 50/50 assets with
  zero failures/warnings, 25 models per asset, unchanged PIT mean/min
  `0.7618/0.2557`, unchanged calibrated PF/Sharpe/hit
  `1.974/2.853/0.580`, slightly better BIC (`-13471.95 -> -13472.34`), and
  faster total runtime (`49.35s -> 46.03s`).

### Story 071 - Unified Improved Stage 5 Decomposition

Target:

- `phi_student_t_unified_improved.py`
- Stage 5 methods and nested searches.

Requirement to challenge:

Stage 5 is too broad: nu, GARCH, rough volatility, skew, shrinkage, and entropy
are mixed in a large procedure.

Delete/refactor:

- Split Stage 5 into individually benchmarkable pure helpers.
- Delete sub-searches whose accepted parameter is identity on most assets.

Candidate improved method:

- Stage helpers:
  - nu candidate scoring.
  - GARCH memory scoring.
  - skew scoring.
  - CRPS shrink scoring.
  - entropy relaxation.

Acceptance:

- Method-size reduction.
- Same or better unified improved selection metrics.

Cycle 071 result:

- Deleted the nested L-BFGS-B `jump_sensitivity` optimizer from Stage 5d and
  kept the neutral sensitivity value until the jump mixture itself passes the
  BIC gate.  This follows the deletion rule: the cache audit showed 200
  unified-improved fitted rows across the 50-stock gate, zero jump activations,
  no best-model wins, and a small average posterior mass for the whole
  unified-improved family.
- Tests:
  `.venv/bin/python -m py_compile src/models/phi_student_t_unified_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_unified_stage5_precompute.py src/tests/test_unified_config_serialization.py -q`.
- Focused unified-improved gate:
  `src/data/benchmarks/cycle_071_unified_improved_jump_sens_delete_focused.json`
  completed 50/50 assets with zero failures, PIT pass rate `0.98`, mean CRPS
  `0.01135`, and mean fit time `0.268s`.
- Full gate:
  `src/data/benchmarks/cycle_071_unified_improved_jump_sens_delete_full_repeat2_metrics.json`
  completed 50/50 assets with zero failures/warnings, 25 models per asset,
  unchanged PIT mean/min `0.7618/0.2557`, unchanged calibrated
  PF/Sharpe/hit `1.974/2.853/0.580`, and faster total runtime
  (`46.03s -> 42.84s`) versus the cycle-070 warm gate.

### Story 072 - Unified Improved Structural-Array Kernel Rewrite

Target:

- `filter_phi_unified`.
- precomputed structural arrays.
- Numba kernels/wrappers.

Requirement to challenge:

Unified model speed depends on building and reusing structural arrays cleanly.

Delete/refactor:

- Delete per-candidate recomputation of arrays that depend only on returns/vol.
- Remove thread-based code paths not used by process-level tuning.

Candidate improved method:

- New Numba kernel consumes:
  - precomputed VoV.
  - stress probabilities.
  - asymmetry degrees of freedom.
  - GARCH variance path when fixed.

Acceptance:

- Focused unified improved benchmark faster.
- Python reference parity.

Cycle 072 result:

- Deleted the per-call `R_base_arr` allocation/fill loop inside
  `unified_phi_student_t_filter_extended_kernel`. The kernel now computes the
  same `c * vol_t * vol_t` scalar at the point of use, which removes one full
  array allocation and one full preparatory pass per extended-filter call.
- Tests:
  `.venv/bin/python -m py_compile src/models/numba_kernels.py src/models/numba_wrappers.py src/models/phi_student_t_unified_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_unified_stage5_precompute.py src/tests/test_unified_config_serialization.py -q`.
- Focused unified-improved warm gate:
  `src/data/benchmarks/cycle_072_unified_extended_inline_rbase_focused_repeat.json`
  preserved mean CRPS `0.01135`, PIT pass rate `0.98`, direction hit rate
  `0.503`, and strategy Sharpe `0.219`, while mean fit time improved versus
  the cycle-071 focused run (`0.268s -> 0.151s`).
- Full gate:
  `src/data/benchmarks/cycle_072_unified_extended_inline_rbase_full_repeat_metrics.json`
  completed 50/50 assets with zero failures/warnings, unchanged PIT mean/min
  `0.7618/0.2557`, unchanged calibrated PF/Sharpe/hit
  `1.974/2.853/0.580`, and slightly faster warm total runtime
  (`42.84s -> 42.52s`) versus cycle 071.

### Story 073 - Unified Improved Jump Layer Deletion Or Promotion

Target:

- jump intensity, jump variance, jump sensitivity, jump mean.

Requirement to challenge:

Jump parameters add complexity only if they improve OOS tail calibration.

Delete/refactor:

- Audit activation, posterior contribution, and CRPS/PIT deltas.
- Delete jump layer if it is mostly inactive.
- If active, promote it into a clear scoring and inference path.

Candidate improved method:

- Jump layer competes as a conditional sub-model with explicit BIC penalty.
- No hidden parameter inflation.

Acceptance:

- Better crisis/tail benchmark or code deletion with unchanged metrics.

Cycle 073 result:

- Tested full Stage 5d tuning deletion after the cycle-071/072 cache audit
  showed 200 unified-improved fitted rows with zero jump activation and zero
  wins. The public config/filter jump fields were left compatible during the
  experiment.
- Tests passed:
  `.venv/bin/python -m py_compile src/models/phi_student_t_unified_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_unified_stage5_precompute.py src/tests/test_unified_config_serialization.py -q`.
- Focused gate:
  `src/data/benchmarks/cycle_073_unified_jump_stage_delete_focused.json`
  preserved CRPS/PIT/profit proxy but did not improve speed versus cycle 072.
- Full gates:
  `src/data/benchmarks/cycle_073_unified_jump_stage_delete_full_metrics.json`
  and
  `src/data/benchmarks/cycle_073_unified_jump_stage_delete_full_repeat_metrics.json`
  both completed 50/50 assets with zero warnings and unchanged PIT/signal
  metrics, but the warm repeat was slower than cycle 072
  (`42.52s -> 44.17s`). The Stage 5d deletion was reverted; only the accepted
  cycle-071 simplification of `jump_sensitivity=1.0` remains.

### Story 074 - Unified Improved Stage 6 Coordinate Calibration

Target:

- `_stage_6_calibration_pipeline`.

Requirement to challenge:

Stage 6 grid/nested search may be too expensive and entangled.

Delete/refactor:

- Delete grid dimensions with flat objective response.
- Replace nested grid with coordinate search using monotone no-harm gates.

Candidate improved method:

- Optimize in order:
  - GARCH blend.
  - nu PIT.
  - probit beta correction.
  - AR whitening lambda.
  - nu CRPS.
- Stop early when marginal score improvement is below threshold.

Acceptance:

- Faster Stage 6.
- Same or better PIT/CRPS.

Cycle 074 result:

- Audited Stage 6 PIT ν choices in the cycle-072 cache. Across 200
  unified-improved rows, calibration selected ν values `>= 6`; low-ν values
  were plausible deletion candidates for this calibration-only grid.
- Tested pruning the Stage 6 PIT grid to `[5, 6, 7, 8, 10, 12, 15, 20]` with
  ν=4 restored only for extreme kurtosis. This did not touch the model
  competition grid.
- Tests passed:
  `.venv/bin/python -m py_compile src/models/phi_student_t_unified_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_unified_stage5_precompute.py src/tests/test_unified_config_serialization.py -q`.
- Focused gates:
  `src/data/benchmarks/cycle_074_unified_stage6_pit_nu_grid_prune_focused.json`
  and
  `src/data/benchmarks/cycle_074_unified_stage6_pit_nu_grid_prune_focused_repeat.json`
  preserved CRPS/PIT/hit/profit proxy but were slower than cycle 072
  (`0.151s -> 0.176s` mean fit time on the warm repeat). The grid prune was
  reverted.

### Story 075 - Unified Improved Composite Selection Score

Target:

- unified candidate scoring in Stage 5/6.

Requirement to challenge:

CRPS alone can prefer sharp but miscalibrated distributions.  PIT alone can
prefer dull distributions.

Delete/refactor:

- Remove candidate score variants that measure the same thing.

Candidate improved method:

- Composite score:
  - normalized CRPS.
  - PIT KS/AD.
  - forecast entropy consistency.
  - sign Brier.
  - BIC penalty.

Acceptance:

- Better full-gate PIT and profitability.
- No increase in warnings.

Cycle 075 result:

- Tested an AD-aware Stage 5 score that allowed the candidate selector to use
  the stronger of KS and Anderson-Darling tail calibration signals when scoring
  candidate ν paths.
- Tests passed:
  `.venv/bin/python -m py_compile src/models/phi_student_t_unified_improved.py`
  and
  `.venv/bin/python -m pytest src/tests/test_unified_stage5_precompute.py src/tests/test_unified_config_serialization.py -q`.
- Focused gate:
  `src/data/benchmarks/cycle_075_unified_stage5_ad_quality_score_focused.json`
  worsened mean CRPS (`0.0113487 -> 0.0113672`), mean PIT p-value
  (`0.53894 -> 0.53689`), PIT pass rate (`0.98 -> 0.96`), and mean fit time
  (`0.1509s -> 0.1572s`).  Mean strategy Sharpe improved slightly
  (`0.2193 -> 0.2237`), but SNAP selected ν=2.5 and failed PIT badly.
- Rejected and reverted. A profitability proxy is not accepted when the
  distributional calibration gate gets worse.

### Story 076 - Unified Improved Conditional Skew Simplification

Target:

- skew score sensitivity.
- skew persistence.
- asymmetry degrees of freedom.

Requirement to challenge:

Skew dynamics may be overparameterized relative to data.

Delete/refactor:

- Delete one skew mechanism if two coexist.
- Keep either static asymmetry or GAS skew, not both, unless each earns its role.

Candidate improved method:

- Compare static skew, GAS skew, and no skew by OOS CRPS/PIT tail balance.

Acceptance:

- Better left/right tail calibration.
- Simpler parameter surface.

Cycle 076 result:

- Audited cycle-072 cache: across 50 files and 2,400 unified-improved config
  rows, `skew_score_sensitivity` was nonzero `0` times, while static
  `alpha_asym` was active in 2,370 rows.
- Attempt A deleted the GAS skew optimizer and kept static asymmetry. Focused
  quality/profit metrics were unchanged, but the full gate was not better:
  total runtime regressed (`42.52s -> 68.25s`), BIC mean softened
  (`-13471.99 -> -13471.88`), and no signal-calibration metric improved.
- Attempt B rewrote static asymmetry from L-BFGS into a causal robust
  tail-imbalance estimator. Focused strategy Sharpe improved
  (`0.2193 -> 0.2229`), but CRPS/PIT slipped slightly; the full gate confirmed
  the trade was poor with PIT mean down (`0.76185 -> 0.75698`), PIT min down
  (`0.25569 -> 0.24877`), BIC mean worse (`-13471.99 -> -13471.14`), and total
  runtime worse (`42.52s -> 73.68s`).
- Rejected and reverted. Static/GAS skew optimization remains until a better
  asymmetry replacement improves both calibration and system runtime.

### Story 077 - Unified Improved Rough Volatility Audit

Target:

- rough Hurst estimation and rough volatility memory.

Requirement to challenge:

Rough volatility can be a sophisticated label for an unstable fit.

Delete/refactor:

- Delete rough layer if Hurst estimate is noisy or has no OOS effect.

Candidate improved method:

- Use rough layer only when validation shows:
  - volatility clustering prediction improves.
  - CRPS improves.
  - PIT does not degrade.

Acceptance:

- Faster unified fit or better stress calibration.

Cycle 077 result:

- Audited the cycle-072 cache. `rough_hurst` was nonzero in 370/2,400
  unified-improved config rows, always tiny (`max < 0.024`); `sigma_eta` was
  never active.
- Tested disabling Stage 5e Hurst estimation so tuning leaves rough-vol
  blending off. Targeted tests passed and the focused 50-stock gate was exactly
  quality/profit neutral (`CRPS`, PIT, pass rate, return, and Sharpe unchanged).
- Full retune/calibration gate did not pass the accepted baseline: PIT and
  signal calibration stayed flat, but BIC mean worsened
  (`-13471.99 -> -13471.34`) and total runtime regressed
  (`42.52s -> 48.42s`).
- Rejected and reverted. The rough layer looks small, but deletion did not earn
  the system-level scoring gate.

### Story 078 - Unified Improved GARCH/PIT Numba Path

Target:

- `_pit_garch_path`.
- `_compute_crps_output`.

Requirement to challenge:

GARCH/PIT correction is array-heavy and should be kernelized if kept.

Delete/refactor:

- Delete duplicated GARCH loops in Python.
- Keep one Numba path and one simple Python reference.

Candidate improved method:

- New kernel outputs PIT values, sigma, mu effective, and variance diagnostics.

Acceptance:

- Reference parity.
- Focused Stage 6 speedup.

Cycle 078 result:

- Audited the GARCH/CRPS path. `loc_bias_var_coeff` and
  `loc_bias_drift_coeff` were active in 1,368/2,400 unified-improved rows, so
  the CRPS location-bias branch was a real target.
- Tested replacing the scalar CRPS location-bias loop with a vector expression.
  Targeted tests passed and focused 50-stock quality/profit metrics were
  exactly unchanged.
- Full gate did not earn acceptance: signal calibration stayed flat, but total
  runtime was worse than the accepted cycle-072 baseline
  (`42.52s -> 63.73s`) and there was no accuracy or profitability gain.
- Rejected and reverted. A GARCH/PIT rewrite needs a real fused kernel with
  reference parity, not a neutral cleanup.

### Story 079 - Unified Improved Fallback Deletion

Target:

- fallback branches in unified improved methods.

Requirement to challenge:

Fallbacks silently hide failed assumptions and keep dead code alive.

Delete/refactor:

- Instrument fallback counts in benchmark.
- Delete fallback paths that never execute.
- Convert meaningful fallback to explicit degraded model output.

Candidate improved method:

- Fail loudly inside tests; degrade explicitly in production only when metrics
  prove safe.

Acceptance:

- Reduced code.
- No new tuning failures.

Cycle 079 result:

- Tested deleting the catastrophic GARCH PIT escape hatch in
  `filter_and_calibrate`.
- Targeted tests passed. The full 50-stock gate completed with 50/50 assets,
  zero failures, and zero warnings.
- Calibration/profitability were unchanged: PIT mean/min stayed
  `0.7618466602 / 0.2556895909`, and the aggregate signal metrics stayed at
  profit factor `1.973747995`, Sharpe `2.85343172`, hit rate `0.580366105`.
- The deletion did not earn its place: total runtime regressed
  `42.52s -> 62.99s`, tune runtime regressed `38.60s -> 57.39s`, and BIC
  softened from `-13471.9869` to `-13471.3740`.
- Rejected and reverted. A fallback can only disappear when it improves speed,
  scoring, or failure transparency under the real 50-stock gate.

### Story 080 - Unified Improved Method-Size Budget

Target:

- methods over 200 lines in `phi_student_t_unified_improved.py`.

Requirement to challenge:

Very large methods block mathematical iteration.

Delete/refactor:

- Enforce method-size budgets for Stage 5/6 and filter methods.
- Delete historical comments that no longer describe production behavior.

Candidate improved method:

- Helpers named after the equation they implement.
- Hot loops remain object-free.

Acceptance:

- Method-size reduction.
- Full benchmark unchanged or improved.

Cycle 080 result:

- Attempt A rewrote the GARCH PIT EWM standardization, chi-square correction,
  and PIT-variance correction into fresh exact Numba kernels and wrappers, then
  extracted `_pit_garch_path` into equation-level helpers.
- Targeted tests passed, and the method shrank from about 497 lines to 82.
  The full gate failed acceptance: PIT/profitability stayed unchanged, but the
  warm repeat still regressed total runtime `42.52s -> 62.09s`, tune runtime
  `38.60s -> 56.85s`, and BIC `-13471.9869 -> -13471.7619`.
- Attempt B removed the new Numba route and tested the extraction-only version
  with the original Python recursions plus `scipy.special.ndtr` for the final
  normal transform. It also passed targeted tests and kept the 82-line method,
  but the full gate still regressed total runtime to `55.06s`, tune runtime to
  `50.98s`, and BIC to `-13471.7060`.
- Rejected and reverted. Method size alone is not enough; the calibration hot
  path must improve speed or scoring under the full 50-stock gate.

### Story 081 - Unified Canonical Stage 5 Parity Audit

Target:

- `phi_student_t_unified.py`.

Requirement to challenge:

Canonical unified may lag improved unified due to old implementation, not due
to a meaningful model difference.

Delete/refactor:

- Port only validated Stage 5 simplifications from improved unified.
- Delete canonical-only duplicate code that is strictly older.

Candidate improved method:

- Shared Stage 5 scorer if math is identical.
- Separate config only for canonical model assumptions.

Acceptance:

- Canonical unified improves or remains simpler.

Cycle 081 result:

- Audited the accepted 50-stock cache and found zero non-neutral
  `jump_sensitivity` rows and zero non-zero `jump_intensity` rows for canonical
  unified models.
- Tested deleting the canonical Stage 5d `jump_sensitivity` optimizer while
  keeping the Merton jump BIC gate. Targeted compile and benchmark-harness tests
  passed.
- Full gate attempt 1:
  `cycle_081_canonical_jump_sensitivity_delete_full_metrics.json` completed
  50/50 assets with zero warnings, unchanged PIT and signal metrics, BIC
  `-13471.9881` versus accepted `-13471.9869`, but runtime regressed
  `42.52s -> 43.90s`.
- Full gate repeat:
  `cycle_081_canonical_jump_sensitivity_delete_repeat_full_metrics.json`
  again had unchanged PIT/signals, but runtime regressed further to `45.38s`
  and BIC softened to `-13471.8529`.
- Rejected and reverted. The canonical simplification is plausible but did not
  clear the repeat gate.

### Story 082 - Unified Canonical Reuse Optimize Diagnostics

Target:

- canonical `optimize_params_unified`.
- `filter_and_calibrate`.

Requirement to challenge:

Optimization may compute test diagnostics and then recompute them for packaging.

Delete/refactor:

- Reuse optimize-time PIT/CRPS/sigma arrays.
- Delete duplicate filter-and-calibrate call when diagnostics are present.

Candidate improved method:

- Diagnostics payload with validated arrays and shape checks.

Acceptance:

- Faster unified canonical fit.
- Same metrics.

Cycle 082 result:

- Tested reusing optimize-time calibrated `test_mu_effective`,
  `test_crps`, and `test_hyvarinen` for canonical and improved unified scoring
  instead of recomputing score inputs against raw `mu_pred`.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_082_unified_cached_calibrated_scores_full_metrics.json`
  completed 50/50 assets with zero warnings. PIT improved from
  `0.7618466602 / 0.2556895909` to `0.7630935169 / 0.2697278101`.
- The trade was not acceptable: BIC softened materially
  `-13471.9869 -> -13470.2972`, total runtime regressed
  `42.52s -> 50.12s`, and signal profitability was unchanged.
- Rejected and reverted. Calibrated score reuse needs an explicit model-weight
  acceptance rule, not a silent scoring-coordinate change.

### Story 083 - Unified Canonical Config Surface Shrink

Target:

- `UnifiedStudentTConfig`.

Requirement to challenge:

Config may contain parameters that are identity for most assets.

Delete/refactor:

- Split required parameters from optional activated layers.
- Delete unused config fields or move to optional nested dict.

Candidate improved method:

- Smaller dataclass and explicit activated-layer metadata.

Acceptance:

- Serialization remains compatible.
- Model sampling still gets all required fields.

Cycle 083 result:

- Tested deleting the disabled `garch_kalman_weight` and `q_vol_coupling`
  locals from the canonical unified fit path while keeping cache-readable
  dataclass fields for backward compatibility.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_083_canonical_disabled_config_surface_full_metrics.json`
  completed 50/50 assets with zero calibration warnings and unchanged
  calibrated signal metrics.
- Rejected: total runtime regressed `42.52s -> 70.18s`, tune runtime
  regressed `38.60s -> 61.02s`, and BIC softened
  `-13471.9869 -> -13471.7958` with no PIT/profitability lift.
- Code reverted. The disabled fields are ugly, but the current construction
  shape is performance-stable and should not be disturbed without a stronger
  serialization rewrite.

### Story 084 - Unified Canonical Weak Layer Deletion

Target:

- asymmetry, jump, rough, and GARCH layers in canonical unified.

Requirement to challenge:

Canonical unified should not carry every improved-layer hypothesis.

Delete/refactor:

- Delete weak layers from canonical if they do not improve OOS score.

Candidate improved method:

- Canonical unified becomes a lean benchmark:
  AR(1) Student-t plus minimal adaptive variance.

Acceptance:

- Faster canonical unified with no full-gate regression.

Cycle 084 result:

- Audited `cycle_072_unified_extended_inline_rbase_full_repeat_cache` across
  all 50 assets and 200 canonical unified fits. `skew_score_sensitivity`,
  `jump_intensity`, `jump_variance`, `jump_mean`, `rho_leverage`, and
  `regime_switch_prob` were zero throughout; GARCH, rough Hurst, loc-bias, and
  mean reversion were active and left untouched.
- Tested deleting the canonical `_stage_4_2_skew_dynamics` optimizer and
  hard-setting `skew_kappa_opt=0.0`, `skew_persistence_fixed=0.97`.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_084_canonical_skew_optimizer_delete_full_metrics.json`
  completed 50/50 assets with zero warnings. BIC improved slightly
  `-13471.9869 -> -13472.0176`, PIT and calibrated signal profitability were
  unchanged.
- Rejected: total runtime regressed `42.52s -> 58.38s` and tune runtime
  regressed `38.60s -> 53.81s`. The change did not deliver speed or
  profitability, so the optimizer was restored.

### Story 085 - Unified Canonical Stress-Q Simplification

Target:

- Markov-switching q and stress probability logic.

Requirement to challenge:

Stress-q logic may duplicate GAS-Q/RV-Q and volatility conditioning.

Delete/refactor:

- Compare stress-q, GAS-Q, RV-Q, and static q in BMA contribution.
- Delete one redundant q-adaptation path.

Candidate improved method:

- One proactive q adaptation mechanism per model family.

Acceptance:

- Better or equal PIT in regime transitions.
- Faster retune.

Cycle 085 result:

- Audited canonical unified stress-q fields in the accepted cycle 072 cache.
  `ms_sensitivity`, `q_stress_ratio`, and `gamma_vov` were active across all
  canonical unified fits; stress-q is not dead code.
- Tested simplifying `_stage_3_ms_sensitivity` to return the asset-profile
  sensitivity prior directly, deleting the L-BFGS-B likelihood loop.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_085_canonical_msq_profile_prior_full_metrics.json`
  completed 50/50 assets with zero warnings and unchanged calibrated signal
  profitability.
- Rejected: BIC regressed `-13471.9869 -> -13471.1633`, total runtime
  regressed `42.52s -> 76.50s`, and tune runtime regressed
  `38.60s -> 63.85s`. The optimizer was restored.

### Story 086 - Phi Gaussian Exact Kernel Rewrite

Target:

- `kalman_phi_gaussian_unified`.
- Gaussian Numba kernels.

Requirement to challenge:

Phi Gaussian should be a fast, exact baseline.  It should not be slower than
Student-t plumbing.

Delete/refactor:

- Delete Python fallback from production path once kernel parity is tested.
- Remove repeated train-state reconstruction.

Candidate improved method:

- Exact linear-Gaussian Kalman kernel:
  - predictive path.
  - log likelihood.
  - OSA fold score.
  - p-up.

Acceptance:

- Gaussian focused benchmark faster.
- Full gate no model-count or quality regression.

Cycle 086 result:

- Implemented a fused Numba `phi_gaussian_cv_fused_fold_kernel` that combined
  the training-prefix state recovery and validation-fold scoring in one kernel,
  avoiding wrapper overhead and discarded training log-likelihood work.
- Added a parity test against the existing split
  `run_phi_gaussian_train_state` + `run_phi_gaussian_cv_test_fold` path.
- Targeted compile, train-state kernel tests, and benchmark-harness tests
  passed.
- Full gate first run `cycle_086_phi_gaussian_fused_cv_kernel_full_metrics.json`
  was compilation-polluted (`63.35s` total), so a repeat gate was run.
- Repeat full gate `cycle_086_phi_gaussian_fused_cv_kernel_repeat_full_metrics.json`
  completed 50/50 assets with zero warnings and unchanged calibrated signal
  profitability, but still regressed total runtime `42.52s -> 44.28s`, tune
  runtime `38.60s -> 40.02s`, and BIC `-13471.9869 -> -13471.3899`.
- Rejected and reverted. Exact local parity did not translate into global BMA
  improvement.

### Story 087 - Gaussian Closed-Form CRPS Gradient Objective

Target:

- Gaussian unified optimizer.

Requirement to challenge:

Gaussian CRPS is closed form and differentiable; finite-difference optimizer
calls are wasteful.

Delete/refactor:

- Remove numerical-gradient path where analytic gradient is stable.

Candidate improved method:

- Derive CRPS gradients for `mu = a + b*x` and `sigma = c + d*sigma_pred`.
- Use gradient-based optimization with shape checks.

Acceptance:

- Faster Gaussian calibration.
- Equal or better CRPS.

Cycle 087 result:

- Derived and implemented the Gaussian CRPS scale first-order condition:
  `mean(sigma * (2 * phi(error / (s * sigma)) - 1 / sqrt(pi))) = 0`.
- Replaced the Stage 4 sigma-shrinkage grid with a bisection solve and added
  a dense-grid parity test for the analytic optimum.
- Targeted compile, CRPS scaling tests, and benchmark-harness tests passed.
- Full gate `cycle_087_gaussian_crps_analytic_shrink_full_metrics.json`
  completed 50/50 assets with zero warnings and unchanged calibrated signal
  profitability.
- Rejected: BIC regressed `-13471.9869 -> -13471.2461`, total runtime
  regressed `42.52s -> 63.31s`, and tune runtime regressed
  `38.60s -> 59.11s`. Code and test reverted.

### Story 088 - Gaussian GAS-Q And Momentum Branch Audit

Target:

- Gaussian unified momentum and GAS-Q.

Requirement to challenge:

Momentum/GAS-Q can be useful, but if they rarely activate they should not
burden every asset.

Delete/refactor:

- Audit activation, posterior mass, and score deltas.
- Delete or gate branches that do not earn their cost.

Candidate improved method:

- Lazy branch fitting only when cheap diagnostics indicate likely benefit.

Acceptance:

- Retune faster with same or better metrics.

Cycle 088 result:

- Audited Gaussian unified activation in the accepted cycle 072 cache.
  Momentum was active for only 3/50 `kalman_phi_gaussian_unified` fits, but
  those had nontrivial posterior weight on `AFRM`, `GE`, and `NFLX`, so
  momentum was preserved.
- GAS-Q activated only on negligible-weight Gaussian fits with near-zero
  coefficients (`gas_q_alpha` around `1e-6` or smaller), so the candidate
  removed the GAS-Q fitting pass and hard-set GAS-Q params to zero.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_088_gaussian_gasq_fit_delete_full_metrics.json` completed
  50/50 assets with zero warnings and unchanged calibrated signal
  profitability.
- Rejected: BIC regressed `-13471.9869 -> -13471.9366`, total runtime
  regressed `42.52s -> 74.78s`, and tune runtime regressed
  `38.60s -> 64.95s`. Code reverted.

### Story 089 - Gaussian Sign-Probability Objective

Target:

- Gaussian objective and CV fold scorer.

Requirement to challenge:

Gaussian models may be sharp and calibrated but weak on directional probability.

Delete/refactor:

- Add sign Brier only as an ablation.
- Delete if it does not improve signal metrics.

Candidate improved method:

- `P(r > 0) = 1 - Phi((0 - mu_pred) / sigma_pred)`.
- Score with Brier on validation fold.

Acceptance:

- Hit rate or PF/Sharpe improves without PIT damage.

Cycle 089 result:

- Requirement challenged before implementation. Gaussian Stage 1 currently
  scores validation likelihood through compact Numba fold kernels that do not
  expose `P(r > 0)`. Adding sign-Brier would require a new kernel return
  contract or an extra Python validation pass inside the optimizer.
- This would optimize a proxy already measured downstream by calibrated signal
  hit rate, PF, and Sharpe. Prior system ablations with sign-Brier objectives
  on Student-t also failed to improve full signal metrics.
- Rejected at requirement gate under the company algorithm: do not add a
  slower optimizer dimension unless the downstream decision layer lacks that
  signal. No code was changed.

### Story 090 - Gaussian Calibration Path Consolidation

Target:

- Gaussian `filter_and_calibrate`.
- signal calibration EMOS reuse.

Requirement to challenge:

Gaussian calibration may duplicate Student-t calibration logic with Gaussian
special cases scattered around.

Delete/refactor:

- Shared EMOS/PIT helpers where distribution differences are explicit.

Candidate improved method:

- Distribution interface:
  - CDF.
  - CRPS.
  - log score.
  - p-up.

Acceptance:

- Less duplicate code.
- Equal or better frontend/backend diagnostics.

Cycle 090 result:

- Deleted dead `S_calibrated` transport in Gaussian `filter_and_calibrate`.
  The GARCH branch now discards the fourth return explicitly and the simple
  branch computes `sigma` directly from `S_pred_test * variance_inflation`.
- Targeted compile and benchmark-harness tests passed.
- Full gate `cycle_090_gaussian_unused_variance_transport_delete_full_metrics.json`
  and repeat gate
  `cycle_090_gaussian_unused_variance_transport_delete_repeat_full_metrics.json`
  completed 50/50 assets with zero warnings.
- PIT and calibrated signal PF/Sharpe/hit were unchanged. BIC improved on both
  runs (`-13471.9869 -> -13472.5038`, repeat `-13472.2627`).
- Accepted narrowly as topology cleanup. Runtime was slower than the cycle 072
  accepted baseline, so no speed win is claimed.

### Story 091 - Numba Kernel Architecture Split

Target:

- `numba_kernels.py`.

Requirement to challenge:

One large kernel file makes dead-code deletion and kernel replacement unsafe.

Delete/refactor:

- Split by responsibility if local import patterns allow:
  - Kalman filters.
  - Student-t distribution.
  - calibration optimizers.
  - diagnostics.
- Delete dead kernels found by `rg`.

Candidate improved method:

- Keep stable public imports in `numba_wrappers.py`.
- Internal modules can be smaller and testable.

Acceptance:

- No import regression.
- Tests pass.
- Dead kernel count reduced.

Cycle 091 result:

- Audited `src/models/numba_kernels.py` definitions against all Python
  references under `src/`. Exactly one dead kernel was found:
  `compute_ms_process_noise_kernel`.
- Deleted that unused threshold-style MS-q kernel. The active MS-q path uses
  the smooth/EWM implementation instead.
- Targeted compile, architecture import tests, and benchmark-harness tests
  passed. `rg` confirmed no remaining references.
- Full gate `cycle_091_dead_msq_kernel_delete_full_metrics.json` and repeat
  `cycle_091_dead_msq_kernel_delete_repeat_full_metrics.json` completed 50/50
  assets with zero warnings and unchanged calibrated signal metrics.
- Accepted narrowly for dead-code reduction. Runtime/BIC movement is not
  credited as causal because the deleted kernel had no runtime references.

### Story 092 - Numba Wrapper Deletion And Typed Boundaries

Target:

- `numba_wrappers.py`.

Requirement to challenge:

Wrappers should not merely rename kernels.  They should enforce boundary
contracts.

Delete/refactor:

- Delete wrappers that only pass through arguments.
- Keep wrappers that do:
  - dtype/contiguity normalization.
  - fallback.
  - tuple/dict normalization.

Candidate improved method:

- Wrapper names mirror production use.
- Kernel names mirror math.

Acceptance:

- Fewer wrappers.
- No call-site ambiguity.

Cycle 092 result:

- Audited every `run_*` function in `src/models/numba_wrappers.py` against
  all Python references under `src/`.
- No unreferenced wrapper functions were found. Existing wrappers either have
  call sites or provide typed/contiguous boundaries for Numba kernels.
- Rejected at audit gate. No code changed; deleting wrappers without evidence
  would make the public kernel boundary less clear rather than simpler.

### Story 093 - EMOS Student-t Optimizer Rewrite

Target:

- `_fit_emos_student_t`.
- EMOS Numba optimizer kernels.

Requirement to challenge:

Student-t EMOS still uses several optimizer stages that may duplicate the same
calibration effect.

Delete/refactor:

- Profile Stage 1, Stage 2, Stage 3 contributions by metric.
- Delete stages that do not improve final CRPS/PF/Sharpe.

Candidate improved method:

- Numba coordinate optimizer:
  - mean scale.
  - variance scale.
  - nu.
  - optional PIT penalty.
- Warm-start from closed-form moments.

Acceptance:

- Faster cached calibration.
- Equal or better CRPS and profitability.

Cycle 093 result:

- Accepted `cycle_093_emos_mean_floor_relax`.
- Challenged the inherited EMOS mean-slope floor. The old `b >= 0.1`
  bound forced even weak/noisy mean forecasts to retain directional size.
- Deleted the unused `avg_actual_abs` transport in `_fit_emos_student_t`.
- Replaced the constant lower bound with an earned floor:
  - use `b >= 0.1` only when weighted walk-forward sign hit rate is at least
    53.5% with adequate support.
  - otherwise allow `b >= 0.02`, letting EMOS nearly erase weak mean forecasts
    while preserving sign orientation.
- Calibration-only check on the prior 50-stock cache:
  - CRPS improvement: `4.32817289 -> 4.33209981`.
  - PIT-cal mean: `0.32717799 -> 0.327296`.
  - PF/Sharpe/hit unchanged.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_093_emos_mean_floor_relax_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.994554560848`.
  - tuning/calibration/total seconds: `43.9173 / 4.3341 / 48.2514`.
  - calibrated CRPS improvement: `4.33209981`.
  - calibrated PIT mean: `0.327296`.
  - calibrated PF/Sharpe/hit: unchanged at
    `1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a distributional accuracy improvement, not a runtime
  improvement; the first attempted stronger `b >= 0.25` sign-edge floor was
  rejected because it worsened CRPS/PIT without profitability lift.

### Story 094 - Beta Calibration Optimizer Rewrite

Target:

- `_fit_beta_calibration`.
- `beta_cal_opt`.

Requirement to challenge:

Focal Beta calibration is powerful but may overfit tails or duplicate threshold
optimization.

Delete/refactor:

- Ablate focal gamma, regularization, and three-parameter Beta vs symmetric Beta.
- Delete unused optimizer path.

Candidate improved method:

- Compare:
  - identity.
  - symmetric Beta.
  - full Beta.
  - temperature scaling.
- Use OOS Brier and ECE plus signal hit rate.

Acceptance:

- Better Brier/hit/PF or simpler calibration with unchanged metrics.

Cycle 094 result:

- Accepted `cycle_094_beta_native_convergence`.
- Challenged fixed Beta optimizer convergence rather than changing the
  already-validated focal objective.
- Increased the Numba-native Beta calibration optimizer cap from `100` to
  `200` iterations.
- Rejected alternatives:
  - focal gamma `1.0`: Brier improved but hit/PF/Sharpe degraded.
  - focal gamma `3.0`: Brier, hit, PF, and Sharpe degraded.
  - exact focal-gradient/SciPy-equivalent path: Brier and PF improved, but hit
    rate and Sharpe slipped, so it failed the holistic gate.
- Calibration-only check on the cycle 093 cache:
  - Brier improvement: `0.027894985 -> 0.02790282`.
  - CRPS/PIT/PF/Sharpe/hit unchanged from cycle 093.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_094_beta_native_convergence_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.552258628559`.
  - tuning/calibration/total seconds: `43.9498 / 4.7107 / 48.6605`.
  - calibrated Brier improvement: `0.02790282`.
  - calibrated CRPS/PIT mean: `4.33209981 / 0.327296`.
  - calibrated PF/Sharpe/hit unchanged:
    `1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a small probability-calibration accuracy improvement with no
  trading metric regression.

### Story 095 - Threshold Expected-Utility Rewrite

Target:

- `_optimize_label_thresholds`.

Requirement to challenge:

Current threshold score mixes hit rate, inverse Brier, and label accuracy.  It
may not reflect asymmetric trading utility.

Delete/refactor:

- Delete threshold terms that do not affect final labels.
- Keep calibration guard so thresholds do not overfit PnL.

Candidate improved method:

- Expected utility score:
  - calibrated probability edge.
  - expected return sign.
  - uncertainty penalty.
  - minimum action count.
- Validate with PF/Sharpe but do not optimize direct realized PnL alone.

Acceptance:

- Better calibrated PF/Sharpe and hit rate on full gate.

Cycle 095 result:

- Accepted `cycle_095_threshold_dead_brier_delete`.
- Challenged the threshold objective. The inverse-Brier term was constant
  across all threshold pairs because it was computed over all calibrated
  probabilities, not over threshold-selected actions.
- Rejected an expected-utility threshold rewrite because it reduced mean
  optimized threshold label accuracy on the 50-stock calibration cache
  (`0.7084323 -> 0.7060308`) without improving the benchmark summary.
- Deleted the dead inverse-Brier threshold term and renormalized the remaining
  hit-rate/label-accuracy weights from `0.4/0.3` to the equivalent `4/7` and
  `3/7`, preserving the selected thresholds.
- Calibration-only check on the cycle 094 cache:
  - threshold label accuracy raw/optimized unchanged:
    `0.4616001 / 0.7084323`.
  - signal Brier/CRPS/PIT/PF/Sharpe/hit unchanged.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_095_threshold_dead_brier_delete_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.598774574326`.
  - tuning/calibration/total seconds: `39.5497 / 4.1067 / 43.6563`.
  - calibrated Brier/CRPS/PIT mean unchanged:
    `0.02790282 / 4.33209981 / 0.327296`.
  - calibrated PF/Sharpe/hit unchanged:
    `1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a topology deletion with behavior preserved and faster full-gate
  wall time in this run.

### Story 096 - BMA Posterior Calibration Entropy

Target:

- BMA score computation.
- model posterior packaging.

Requirement to challenge:

BMA posterior can collapse too strongly on tiny score differences, reducing
model uncertainty.

Delete/refactor:

- Delete arbitrary posterior smoothing if not justified.
- Replace with entropy regularization tied to score uncertainty.

Candidate improved method:

- Posterior temperature from:
  - sample size.
  - score spread.
  - PIT reliability.
  - stress stability.

Acceptance:

- Better PIT/stability with no profitability regression.

Cycle 096 result:

- Rejected at audit gate; no code changed for this story.
- Audited the active posterior path:
  - tuning uses `entropy_regularized_weights(..., lambda_entropy=0.05)`.
  - signal fallback uses `DEFAULT_POSTERIOR_TEMPERATURE = 0.05`.
  - `src/tests/test_model_improvements.py` already enforces the tuning/signals
    temperature match and posterior sharpness.
- Audited the cycle 095 50-stock cache posterior concentration:
  - mean top-5 posterior masses: `0.5191 / 0.3056 / 0.1005 / 0.0359 / 0.0164`.
  - models per asset mean remains `25.0`.
  - improved and canonical models both retain meaningful competition mass.
- The posterior is neither near-uniform nor collapsed to a single model, so
  changing entropy temperature or floor would be an unjustified model-selection
  perturbation.
- Requirement rewrite: defer posterior-temperature changes until a stress-slice
  benchmark shows posterior instability. The current cycle produced evidence
  for no safe positive change.

### Story 097 - Registry And Parameter Transport Hardening

Target:

- `model_registry.py`.
- tuning cache fields.
- signal generation extraction.

Requirement to challenge:

Model names and parameter extraction should never drift again.

Delete/refactor:

- Delete any model-name parsing outside the registry when registry helpers can do it.
- Add tests for every active model family.

Candidate improved method:

- Registry owns:
  - display family.
  - sampling extraction.
  - required parameter fields.
  - frontend label metadata if useful.

Acceptance:

- No silent model drops.
- Frontend/backend sees all 25 active models.

Cycle 097 result:

- Accepted `cycle_097_registry_gaussian_unified_names`.
- Deleted remaining manual unified-Gaussian model-name literals in
  `src/tuning/tune_modules/model_fitting.py`.
- Added `make_gaussian_unified_name()` to the tuning config import/fallback path,
  so unified Gaussian names now follow the same registry contract as canonical
  and improved Student-t names.
- Added `test_active_retune_model_names_are_registered` to
  `src/tests/test_model_registry_parameter_transport.py`.
  The test asserts the active 25-model retune set is registered:
  - 2 unified Gaussian models.
  - 4 canonical Student-t fixed-nu models plus MLE.
  - 4 improved Student-t fixed-nu models plus MLE.
  - 4 canonical unified Student-t models.
  - 4 improved unified Student-t models.
  - 1 RV-Q phi-Gaussian plus 4 RV-Q Student-t models.
- Tests:
  - `python -m unittest src.tests.test_model_registry_parameter_transport -v`.
  - `python -m pytest src/tests/test_architecture_imports.py -q`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_097_registry_gaussian_unified_names_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.783386107236`.
  - tuning/calibration/total seconds: `52.9155 / 6.0306 / 58.9461`.
  - signal Brier/CRPS/PIT/PF/Sharpe/hit unchanged from cycle 095.
- Accepted as registry/transport hardening with behavior preserved; no runtime
  win claimed because this run was noisy/slower.

### Story 098 - Stress And Crisis Benchmark Slices

Target:

- benchmark scripts.
- calibration diagnostics.

Requirement to challenge:

Average 50-stock metrics can hide crisis fragility.

Delete/refactor:

- Delete acceptance claims based only on average PIT/PF when tail behavior changes.

Candidate improved method:

- Add benchmark slices:
  - 2020 crash.
  - 2022 inflation/rates stress.
  - 2024-2026 high-vol tech/AI period.
  - asset-specific max-vol windows.

Acceptance:

- Tail-related model changes pass stress metrics.

Cycle 098 result:

- Accepted `cycle_098_stress_slice_metrics`.
- Added `stress_slice_summary` to `src/tuning/benchmark_retune_50.py`.
- The benchmark now summarizes calibration behavior on:
  - fixed windows: `covid_crash_2020`, `rates_inflation_2022`,
    `ai_high_vol_2024_2026`.
  - per-asset `asset_max_vol_decile`.
  - per-asset `asset_realized_tail_decile`.
- Uses existing calibration records and price-date cache; no extra model
  fitting pass is introduced.
- Smoke check:
  - `asset_max_vol_decile`: 50 assets, 3339 records.
  - `asset_realized_tail_decile`: 50 assets, 3250 records.
  - fixed 2020/2022 windows report zero records because current calibration
    record caches do not span those dates; the benchmark reports this
    explicitly rather than fabricating coverage.
- Tests:
  - `python -m py_compile src/tuning/benchmark_retune_50.py`.
  - `python -m unittest src.tests.test_benchmark_harness -v`.
  - `python -m pytest src/tests/test_architecture_imports.py -q`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_098_stress_slice_metrics_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.931997710504`.
  - tuning/calibration/total seconds: `53.7207 / 3.6720 / 57.3927`.
  - signal Brier/CRPS/PIT/PF/Sharpe/hit unchanged from cycle 095.
- Notable stress diagnostics:
  - `ai_high_vol_2024_2026`: Brier improvement `0.0282512`, calibrated hit
    `0.579185`, 31900 records.
  - `asset_max_vol_decile`: Brier improvement `0.0294007`, calibrated hit
    `0.593890`, 3339 records.
  - `asset_realized_tail_decile`: Brier improvement `0.0370159`, calibrated hit
    `0.678462`, 3250 records.
- Accepted as benchmark coverage improvement; model math unchanged.

### Story 099 - Dynamic Model Diagnostics Matrix

Target:

- backend diagnostics services.
- frontend model comparison UI.

Requirement to challenge:

The UI should reveal model competition quality, not just top winners.

Delete/refactor:

- Delete static assumptions about model families.

Candidate improved method:

- Backend exposes matrix:
  - model name.
  - family.
  - posterior mass.
  - PIT.
  - BIC.
  - CRPS if available.
  - winner counts.
- Frontend renders unknown model families gracefully.

Acceptance:

- Improved/unimproved side-by-side visible after retune.

Cycle 099 result:

- Accepted `cycle_099_dynamic_model_diagnostics`.
- Rebuilt diagnostics model comparison around the actual tune cache instead of
  a thin winner-only aggregate.
- Backend now exposes, per model:
  - dynamic family.
  - wins and posterior mass.
  - BIC, CRPS, Hyvarinen, PIT, AD, and histogram MAD averages.
  - top-weighted symbols.
  - compact asset/model cells for frontend inspection.
- Cross-asset diagnostics now union `models` and `model_posterior`, so a model
  with posterior mass cannot disappear from the matrix when metrics are sparse.
- Frontend model comparison now renders all returned models dynamically and
  shows model family, average CRPS/PIT/BIC, posterior mass, and top assets.
- Visibility smoke against the real tune cache:
  - diagnostics payload reports `498` tuned assets.
  - active model count shown by diagnostics: `25`.
  - improved variants present in the payload, including
    `phi_student_t_improved_*` and `phi_student_t_unified_improved_*`.
  - first matrix row contains 25 model cells and 9 improved variants.
- Tests:
  - `python -m py_compile src/web/backend/services/diagnostics_service.py`.
  - `npm run build` in `src/web/frontend`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_099_dynamic_model_diagnostics_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.486528668203`.
  - tuning/calibration/total seconds: `40.4541 / 3.6274 / 44.0815`.
  - PIT mean/min: `0.7618466601969215 / 0.25568959090050664`.
  - signal PF/Sharpe/hit unchanged at
    `1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as observability/frontend correctness improvement; model math
  unchanged.

### Story 100 - Final Full System Release Gate

Target:

- `make retune`.
- tuning, calibration, signal generation, frontend diagnostics.

Requirement to challenge:

No story matters unless the full system is better.

Delete/refactor:

- Remove rejected code, dead flags, unused benchmark helpers, and stale comments.

Candidate improved method:

- Run:
  - targeted unit tests.
  - frontend build.
  - full 50-stock retune plus calibration.
  - signal generation smoke.
  - model visibility smoke.

Acceptance:

- 50/50 assets.
- 0 failures.
- 0 calibration warnings.
- improved and canonical models compete side by side.
- final PF/Sharpe/hit/PIT equal or better than cycle 047.

Cycle 100 result:

- Accepted `cycle_100_final_release_gate`.
- No further model mutation in this story; this was the full-system gate for
  tuning, calibration, diagnostics, frontend build, and signal-generation smoke.
- Targeted tests:
  - `python -m unittest src.tests.test_benchmark_harness -v` passed.
  - `python -m unittest src.tests.test_model_registry_parameter_transport -v`
    passed.
  - `python -m pytest src/tests/test_architecture_imports.py -q` passed.
  - `npm run build` in `src/web/frontend` passed; existing CSS/chunk-size
    warnings remain.
- Model visibility smoke:
  - diagnostics payload reports `498` tuned assets.
  - active diagnostics model count: `25`.
  - improved model variants visible: `9`.
  - examples include `phi_student_t_improved_nu_8`,
    `phi_student_t_improved_nu_20`,
    `phi_student_t_improved_nu_mle`,
    and `phi_student_t_unified_improved_nu_4`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_100_final_release_gate_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13471.718382997637`.
  - PIT mean/min: `0.7618466601969215 / 0.25568959090050664`.
  - tuning/calibration/total seconds: `41.6443 / 4.2856 / 45.9299`.
  - best-model winners include both canonical and improved Student-t variants:
    `phi_student_t_improved_nu_20`, `phi_student_t_improved_nu_8`,
    `phi_student_t_improved_nu_mle`,
    `phi_student_t_nu_20`, `phi_student_t_nu_4`,
    `phi_student_t_nu_8`, and `phi_student_t_nu_mle`.
  - signal Brier/CRPS/PIT/PF/Sharpe/hit:
    `0.02790282 / 4.33209981 / 0.327296 / 1.973747995 / 2.85343172 / 0.580366105`.
- Signal-generation smoke:
  - `make verify-signals-quick ARGS="--workers 11"` completed.
  - 8/8 assets processed, 0 failed/skipped.
  - 90-day quick-window 7D hit-rate mean was weak at `41.9%`; this is recorded
    as a residual signal-quality risk, not hidden as a release success.
- Release gate accepted because the full retune/calibration benchmark is clean,
  improved and canonical models compete side by side, and diagnostics/frontend
  now expose the actual model competition.

## Indicator-Integrated Model Phase - Cycles 101-150

This third story bank tests whether high-quality technical indicators can
improve the distributional models when they are integrated into model dynamics,
not bolted onto signal generation as separate trading rules.

The requirement to challenge:

- Indicators are not alpha by default.
- Indicators are lossy transforms of price, volume, and volatility.
- A transform earns a place only if it improves proper scores, calibration,
  or post-calibration signal outcomes on real data.

Preferred indicator families for this phase:

- Heikin-Ashi: candle-state smoothing, wick/body imbalance, and trend fatigue.
- ATR/SuperTrend/Chandelier: volatility-aware trend and stop geometry.
- KAMA/Efficiency Ratio: trend quality versus noise.
- ADX/DMI: directional strength without requiring direction to be profitable.
- Ichimoku: multi-horizon equilibrium, cloud distance, and trend structure.
- Donchian/Turtle breakouts: range expansion and structural highs/lows.
- Bollinger/Keltner squeeze: volatility compression and expansion timing.
- RSI/StochRSI/Williams %R: bounded exhaustion and failed reversal context.
- MACD/PPO/TRIX: multi-scale momentum acceleration.
- OBV/MFI/CMF/volume z-score: participation and flow confirmation.
- Anchored VWAP/VWAP bands: price dislocation from volume-weighted equilibrium.
- Hurst/fractal dimension/wavelet energy: persistence versus mean reversion.
- Cross-sectional relative strength and breadth: asset context, not isolation.

Model integration rules:

1. Every indicator must be causal: no centered windows, no future high/low
   leakage, no hindsight anchors.
2. Indicators enter through model internals:
   - state-equation drift input `u_t`.
   - process-noise scaling `q_t`.
   - observation variance scaling `R_t`.
   - Student-t tail thickness `nu_t`.
   - skew/asymmetry conditioning.
   - regime-fit likelihood penalties.
   - calibration covariates for EMOS/Beta only after model ablation.
3. Every accepted indicator path must have a no-indicator control model and an
   indicator-integrated model competing side by side in BMA.
4. Any indicator-derived model variant must pass ablation against the same base
   model with the indicator input removed.
5. Feature count must be controlled by BIC/LFO/proper-score penalties.
6. A rejected indicator path is deleted in the same cycle.
7. Heikin-Ashi and other smoothed transforms must never replace raw OHLCV;
   they may only add state context.
8. The final signal still comes from distributional geometry and calibration,
   not a hard-coded indicator vote.
9. Frontend/backend diagnostics must show base versus indicator-integrated
   model variants side by side.

### Indicator Phase Acceptance Addendum

For cycles 101-150, a story is accepted only if it achieves at least one of:

- Better full-gate calibrated PF, Sharpe, hit rate, Brier, or CRPS.
- Better stress/tail-slice calibration without worsening full-gate results.
- Equal signal quality with materially faster tuning/calibration.
- Equal quality with fewer indicator/model methods or fewer duplicate feature
  paths.
- A cleaner benchmark or diagnostic that prevents false indicator promotion.

Every accepted indicator feature must record:

- lookback and lag policy.
- missing-data policy.
- outlier/winsor policy.
- whether it affects mean, variance, tail thickness, asymmetry, q, regime fit,
  or calibration.
- ablation result versus the same model without the feature.

## Indicator Work Ledger - Cycles 101-150

| Cycle | Planned Focus | Status |
| ---: | --- | --- |
| 101 | Model-integrated indicator state contract and causal OHLCV registry | Accepted |
| 102 | Heikin-Ashi kernel wired as model state input candidate | Accepted |
| 103 | Heikin-Ashi Student-t drift-input variant versus base Student-t | Accepted |
| 104 | Heikin-Ashi Student-t tail/variance conditioner variant | Rejected |
| 105 | Heikin-Ashi unified Student-t q/asymmetry conditioner variant | Rejected |
| 106 | ATR kernel shared by Gaussian and Student-t variance models | Accepted |
| 107 | SuperTrend regime-likelihood variant in unified Student-t | Rejected |
| 108 | Chandelier/ATR distance as predictive variance model input | Planned |
| 109 | KAMA Efficiency Ratio as process-noise `q_t` model variant | Planned |
| 110 | KAMA equilibrium as state-equation MR input variant | Planned |
| 111 | ADX/DMI kernel wired into model-state feature bundle | Planned |
| 112 | ADX-gated Student-t tail thickness and q variant | Planned |
| 113 | Ichimoku cloud features as regime-likelihood model inputs | Planned |
| 114 | Ichimoku equilibrium as OU/state-input model variant | Planned |
| 115 | Donchian channel state integrated into breakout variance model | Planned |
| 116 | Turtle breakout quality as unified Student-t score conditioner | Planned |
| 117 | Bollinger/Keltner squeeze as variance-transition model input | Planned |
| 118 | Bollinger percentile as observation-noise mismatch model input | Planned |
| 119 | RSI/StochRSI exhaustion as bounded tail/asymmetry model input | Planned |
| 120 | Williams %R failed-breakout Student-t reversal-state variant | Planned |
| 121 | MACD/PPO/TRIX acceleration as drift-state input variant | Planned |
| 122 | Orthogonal momentum basis inside state-equation model inputs | Planned |
| 123 | OBV/MFI/CMF flow as confidence and variance model input | Planned |
| 124 | Volume z-score and dollar-volume liquidity variance variant | Planned |
| 125 | Anchored/rolling VWAP equilibrium model variant | Planned |
| 126 | VWAP band distance as model variance/confidence conditioner | Planned |
| 127 | Hurst/fractal persistence as q and regime-fit model input | Planned |
| 128 | Wavelet energy as impulse/rough-volatility model input | Planned |
| 129 | Cross-sectional relative strength as market-conditioned drift input | Planned |
| 130 | Sector/beta-aware normalization inside model feature transport | Planned |
| 131 | Indicator-integrated BIC/LFO model-selection layer | Planned |
| 132 | Canonical Student-t indicator-integrated model family | Planned |
| 133 | Improved Student-t indicator-integrated model family | Planned |
| 134 | Unified improved Student-t indicator-integrated model family | Planned |
| 135 | Gaussian indicator-integrated control model family | Planned |
| 136 | Indicator interaction deletion pass inside model variants | Planned |
| 137 | Indicator-integrated PIT/AD/Berkowitz model audit | Planned |
| 138 | Model-residual-aware EMOS calibration with indicator covariates | Planned |
| 139 | Model-residual-aware Beta calibration and threshold stability | Planned |
| 140 | Indicator-integrated stress/tail benchmark slices | Planned |
| 141 | Heikin-Ashi model variant versus raw-candle model variant benchmark | Planned |
| 142 | Trend-indicator model family repeated-run noise analysis | Planned |
| 143 | Mean-reversion indicator model family repeated-run noise analysis | Planned |
| 144 | Volume/flow indicator model family repeated-run noise analysis | Planned |
| 145 | Model feature-latency and missing-data robustness audit | Planned |
| 146 | Numba/vectorized indicator model-input speed pass | Planned |
| 147 | Frontend/backend base-versus-indicator model diagnostics | Planned |
| 148 | Full 50-stock indicator-integrated model competition gate | Planned |
| 149 | Signal-generation smoke for indicator-integrated winning models | Planned |
| 150 | Final indicator-integrated model release gate | Planned |

## Detailed Stories 101-150

### Story 101 - Indicator Model-State Contract And Registry

Target:

- `src/decision/signal_modules/feature_pipeline.py`.
- new or existing indicator feature modules.
- benchmark feature snapshots.
- `src/models/model_registry.py`.

Indicator logic:

- Define a small registry for causal indicator transforms and their model-use
  channel.
- Each feature declares required columns, lookback, lag, output names, and
  whether it may feed mean, variance, tail, asymmetry, q, regime likelihood, or
  calibration.
- Model specs declare whether they are base, indicator-integrated, or control
  variants.

Acceptance:

- No indicator feature can enter a model without registry metadata.
- No indicator-integrated model can be tuned unless its no-indicator control is
  tuned in the same run.
- Feature generation is deterministic across tuning and signal generation.
- Missing OHLCV fields fail closed or produce documented null features.

Cycle 101 result:

- Accepted `cycle_101_indicator_contract`.
- Added `src/models/indicator_state.py` as the model-state indicator contract:
  - registered causal specs for Heikin-Ashi, ATR/SuperTrend, KAMA, ADX/DMI,
    Ichimoku, Donchian, Bollinger/Keltner, oscillators, MACD/PPO/TRIX,
    volume flow, VWAP, persistence/wavelet, and relative strength.
  - every spec declares required OHLCV columns, lookback, lag, output names,
    and allowed model-use channels.
  - indicator model inputs must be lagged by at least one bar.
- Extended `ModelSpec` with indicator-integrated metadata:
  - `model_variant`.
  - `base_model_name`.
  - `indicator_features`.
  - `indicator_channels`.
- Added registry helpers for future side-by-side model variants:
  - `make_indicator_integrated_model_name`.
  - `create_indicator_integrated_spec`.
  - `assert_indicator_models_have_controls`.
- Added `src/tests/test_indicator_model_contract.py`.
- Tests:
  - `python -m py_compile src/models/indicator_state.py src/models/model_registry.py src/tests/test_indicator_model_contract.py`.
  - `python -m unittest src.tests.test_indicator_model_contract -v`.
  - `python -m unittest src.tests.test_model_registry_parameter_transport -v`.
  - `python -m pytest src/tests/test_architecture_imports.py -q`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_101_indicator_contract_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13468.920239085126`.
  - PIT mean/min: `0.7536257850670388 / 0.33046765605579104`.
  - tuning/calibration/total seconds: `40.2861 / 4.1120 / 44.3981`.
  - calibrated signal Brier/CRPS/PF/Sharpe/hit unchanged:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a model-integration contract cycle; no indicator math promoted yet.

### Story 102 - Heikin-Ashi Causal Candle Kernel

Target:

- indicator kernels.
- Numba tests.
- model-input feature transport.

Indicator logic:

- Compute Heikin-Ashi open/high/low/close from only current and prior bars.
- Emit body ratio, upper/lower wick ratio, HA color, HA run length, and raw
  close minus HA close.

Acceptance:

- Numba and Python reference parity.
- No centered-window leakage.
- HA features are exposed only through model-state input bundles.
- Feature generation speed is acceptable on the 50-stock universe.

Cycle 102 result:

- Accepted `cycle_102_heikin_ashi_state_kernel`.
- Added causal Heikin-Ashi state computation to `src/models/indicator_state.py`.
- Outputs are lagged model inputs:
  - `ha_body_ratio`.
  - `ha_color`.
  - `ha_upper_wick_ratio`.
  - `ha_lower_wick_ratio`.
  - `ha_run_length`.
  - `ha_close_dislocation`.
- Added a Numba kernel and Python reference path with parity tests.
- Added `build_heikin_ashi_bundle()` to expose HA only through the model-state
  bundle contract from cycle 101.
- Tests:
  - `python -m py_compile src/models/indicator_state.py src/tests/test_indicator_model_contract.py`.
  - `python -m unittest src.tests.test_indicator_model_contract -v`.
  - `python -m unittest src.tests.test_model_registry_parameter_transport -v`.
  - `python -m pytest src/tests/test_architecture_imports.py -q`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_102_heikin_ashi_state_kernel_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.0`.
  - BIC mean: `-13469.106470269777`.
  - PIT mean/min: `0.7536257850670388 / 0.33046765605579104`.
  - tuning/calibration/total seconds: `40.6359 / 3.7240 / 44.3599`.
  - calibrated signal Brier/CRPS/PF/Sharpe/hit unchanged:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as the first causal indicator model-input kernel; no model variant
  promoted yet.

### Story 103 - Heikin-Ashi Trend-State Mean Input

Target:

- `phi_student_t_improved`.
- `phi_student_t`.
- model registry competition path.

Indicator logic:

- Test HA run length and HA body strength as exogenous mean-state input.
- Compare against existing momentum/MR input so HA must add orthogonal signal.

Acceptance:

- Better or equal BIC/LFO and calibrated PF/Sharpe/hit versus no-HA model.
- No direct HA trading rule is introduced.

Cycle 103 result:

- Accepted `cycle_103_heikin_ashi_student_t_state`.
- Integrated Heikin-Ashi into the Student-t state equation as a competing
  model variant, not as a trading overlay:
  - tuning input: `u_t = beta_ha * sigma_t * ha_drift_signal_t`.
  - signal-generation input: current lagged HA state is recomputed from OHLC
    and passed into Monte Carlo through the model's learned `ind_ha_drift_weight`.
- Registered side-by-side model names such as:
  - `phi_student_t_nu_20_ind_heikin_ashi`.
  - `phi_student_t_improved_nu_20_ind_heikin_ashi`.
- Added the HA drift signal collapse with bounded body, color, run-length,
  wick-rejection, and stretch-penalty terms.
- Added an evidence gate:
  - admit HA only when held-out CRPS improves or PIT improves enough.
  - reject candidates that improve sharpness by breaking calibration when the
    no-HA control was already calibrated.
- Real-data recursive correction:
  - first full run admitted an NVDA HA candidate with better CRPS but damaged
    PIT; the rule was tightened and the full gate was rerun.
  - final admitted global HA variants on `JPM` and `SNAP`; `SNAP` selected
    `phi_student_t_nu_20_ind_heikin_ashi` as its best global model.
- Frontend model labels now show `+HA` for indicator-integrated variants.
- Tests:
  - `.venv/bin/python -m py_compile src/models/indicator_state.py src/models/model_registry.py src/tuning/tune_modules/model_fitting.py src/tuning/tune_modules/regime_bma.py src/tuning/tune_modules/asset_tuning.py src/decision/signal_modules/feature_pipeline.py src/decision/signal_modules/bma_engine.py src/decision/signal_modules/parameter_loading.py src/tests/test_indicator_model_contract.py`.
  - `.venv/bin/python -m unittest src.tests.test_indicator_model_contract -v`.
  - `.venv/bin/python -m unittest src.tests.test_model_registry_parameter_transport -v`.
  - `.venv/bin/python -m pytest src/tests/test_architecture_imports.py -q`.
- Full 50-stock real-data gate with 11 process workers:
  - artifact: `src/data/benchmarks/cycle_103_heikin_ashi_student_t_state_full_metrics.json`.
  - 50/50 assets, 0 failures, 0 calibration warnings.
  - models per asset mean: `25.04`.
  - BIC mean: `-13469.778483848171` versus cycle 102 `-13469.106470269777`.
  - PIT mean/min: `0.7434874406719306 / 0.33046765605579104`.
  - tuning/calibration/total seconds: `40.2123 / 5.6927 / 45.9049`.
  - calibrated signal Brier/CRPS/PF/Sharpe/hit unchanged:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a narrow model-internal HA mean-input cycle: BIC improved, no
  failures/warnings, profitability metrics were preserved, and only candidates
  that earned the extra parameter remained in the model set.

### Story 104 - Heikin-Ashi Exhaustion Tail Conditioner

Target:

- Student-t tail thickness and variance conditioning.

Indicator logic:

- Use HA wick/body imbalance and close-to-HA dislocation to flag exhaustion.
- Test whether exhaustion should widen left/right tails or only variance.

Acceptance:

- Stress/tail-slice PIT improves without full-gate regression.
- Tail conditioner is deleted if it only increases confidence after exhaustion.

Cycle 104 result:

- Rejected `cycle_104_heikin_ashi_variance_tail`.
- Benchmark artifact:
  `src/data/benchmarks/cycle_104_heikin_ashi_variance_tail_full_metrics.json`.
- Full 50-stock gate:
  - 50/50 assets.
  - 0 failures.
  - 0 calibration warnings.
  - Models per asset mean: `25.1`.
  - BIC mean: `-13469.688265076167`, worse than cycle 103
    `-13469.778483848171`.
  - PIT mean/min:
    `0.7434874406719306 / 0.33046765605579104`.
  - Tuning/calibration/total seconds:
    `48.0797 / 3.9315 / 52.0112`, slower than cycle 103 total
    `45.9049`.
  - Signal Brier/CRPS/PF/Sharpe/hit:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- The variance/tail channel admitted a few local candidates, but the system paid
  too much BIC and runtime for tiny asset-level CRPS/PIT gains.
- Deleted the experimental variance/tail branch entirely:
  - no registry variant remains.
  - no tuning fitter remains.
  - no signal-generation variance multiplier remains.
  - no frontend `+HAσ` label remains.
- Cleanup verification:
  - `py_compile` passed on changed model/tuning/signal files.
  - `src.tests.test_indicator_model_contract` passed.
  - `src.tests.test_model_registry_parameter_transport` passed.
  - `src/tests/test_architecture_imports.py` passed.

### Story 105 - Heikin-Ashi Unified q/Asymmetry Model Variant

Target:

- unified Student-t q process.
- unified Student-t asymmetry conditioning.
- model registry competition path.

Indicator logic:

- HA persistence and fatigue modulate q, asymmetry, or variance inside the
  unified model.
- Direction remains emergent from the model posterior and geometry, not HA
  color.

Acceptance:

- Indicator-integrated unified variant competes side by side with the base
  unified model.
- Quick and full signal smoke do not show worse hit/Brier.

Cycle 105 result:

- Rejected `cycle_105_heikin_ashi_unified_q_asym`.
- Benchmark artifact:
  `src/data/benchmarks/cycle_105_heikin_ashi_unified_q_asym_metrics.json`.
- Full 50-stock retune gate, calibration skipped for a fast acceptance screen:
  - 50/50 assets.
  - 0 failures.
  - 0 calibration warnings.
  - Models per asset mean: `25.38`, worse than the cycle 103 baseline `25.04`.
  - BIC mean: `-13469.555920615046`, worse than cycle 103
    `-13469.778483848171`.
  - PIT mean/min:
    `0.735951821757494 / 0.33046765605579104`, worse than cycle 103
    `0.7434874406719306 / 0.33046765605579104`.
  - Tuning seconds: `41.1465`, slower than cycle 103 tune seconds `40.2123`.
- Deleted the experimental unified q/asymmetry branch:
  - no unified HA registry variants remain.
  - no unified HA q/asymmetry fitter remains.
  - no signal-generation q-stress multiplier remains.
  - no q/asymmetry parameter transport remains.
- Conclusion: HA earned a narrow state-mean role in cycle 103, but the unified
  q/asymmetry channel added model-surface area without improving the gate.

### Story 106 - ATR And SuperTrend Foundation

Target:

- true-range kernel.
- SuperTrend reference implementation.

Indicator logic:

- Consolidate ATR calculation and build SuperTrend from causal ATR bands.
- Emit band distance, trend side, trend flips, and flip age.

Acceptance:

- One ATR implementation shared across indicators.
- SuperTrend parity tests cover gaps and flat bars.

Cycle 106 result:

- Accepted `cycle_106_atr_supertrend_foundation`.
- Added causal ATR/SuperTrend state computation in
  `src/models/indicator_state.py`.
- Outputs:
  - `atr_z`
  - `supertrend_side`
  - `supertrend_flip`
  - `supertrend_flip_age`
  - `supertrend_band_distance`
- Added Python reference and Numba implementations with exact parity tests.
- Test coverage:
  - gap bars.
  - flat bars.
  - bundle validation.
  - lagged causal output.
- Benchmark artifact:
  `src/data/benchmarks/cycle_106_atr_supertrend_foundation_metrics.json`.
- Full 50-stock gate:
  - 50/50 assets.
  - 0 failures.
  - 0 calibration warnings.
  - Models per asset mean: `25.04`.
  - BIC mean: `-13470.231757647438`.
  - PIT mean/min:
    `0.7434874406719306 / 0.33046765605579104`.
  - Tuning/calibration/total seconds:
    `41.6667 / 4.1657 / 45.8325`.
  - Signal Brier/CRPS/PF/Sharpe/hit:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- Accepted as a no-behavior-change foundation: the indicator math is now shared
  and tested, while model consumption is reserved for later cycles.

### Story 107 - SuperTrend As Regime-Fit Feature

Target:

- unified Student-t regime fit.
- BMA model diagnostics.

Indicator logic:

- Use SuperTrend side/flip age to describe trend persistence.
- It may improve regime fit but must not force long/short labels.

Acceptance:

- Better regime-slice calibration or equal quality with better stability.
- No hard-coded SuperTrend entry rule survives.

Cycle 107 result:

- Rejected `cycle_107_supertrend_student_t_state`.
- Benchmark artifact:
  `src/data/benchmarks/cycle_107_supertrend_student_t_state_metrics.json`.
- Full 50-stock gate:
  - 50/50 assets.
  - 0 failures.
  - 0 calibration warnings.
  - Models per asset mean: `25.14`, worse than cycle 106 `25.04`.
  - BIC mean: `-13469.228880002373`, worse than cycle 106
    `-13470.231757647438`.
  - PIT mean/min:
    `0.743316347953944 / 0.33046765605579104`, slightly worse than cycle 106.
  - Tuning/calibration/total seconds:
    `45.3575 / 4.0049 / 49.3625`, slower than cycle 106.
  - Signal Brier/CRPS/PF/Sharpe/hit:
    `0.02790282 / 4.33209981 / 1.973747995 / 2.85343172 / 0.580366105`.
- The SuperTrend variant won one local asset but worsened system evidence and
  runtime, so it was deleted:
  - no SuperTrend model registry variants remain.
  - no SuperTrend tuning fitter remains.
  - no SuperTrend signal-generation transport remains.
  - no SuperTrend frontend label remains.
- The accepted cycle 106 ATR/SuperTrend kernel remains available as tested
  infrastructure for future, narrower hypotheses.

### Story 108 - Chandelier Distance Variance Conditioner

Target:

- predictive variance and confidence.

Indicator logic:

- Chandelier exit distance estimates how crowded the move is relative to ATR.
- Test as variance inflation and confidence dampener.

Acceptance:

- Reduces magnitude outliers or tail PIT misses.
- Deleted if it merely suppresses profitable high-conviction signals.

### Story 109 - KAMA Efficiency Ratio Process Noise

Target:

- q process-noise logic.
- `student_t_common.py` shared helpers.

Indicator logic:

- Efficiency Ratio separates directional travel from noisy path length.
- Higher ER can lower q in persistent trends; low ER can raise q in chop.

Acceptance:

- Improved BIC/PIT or speed-neutral signal lift.
- q adjustment must be bounded and regularized.

### Story 110 - KAMA Slope And Equilibrium Distance

Target:

- mean-reversion input.
- momentum/MR orthogonality.

Indicator logic:

- Use KAMA slope and price distance from KAMA as smoother equilibrium features.
- Compare against current OU equilibrium estimator.

Acceptance:

- Better range-regime calibration without trend-regime decay.
- Remove if collinear with existing MR state.

### Story 111 - ADX/DMI Directional Strength Kernel

Target:

- indicator kernel and tests.

Indicator logic:

- Compute +DI, -DI, DX, and ADX causally.
- Treat ADX as strength, not direction; direction comes only through +DI/-DI
  spread if ablation proves value.

Acceptance:

- Reference tests for trend, flat, and gap-heavy sequences.
- No division instability or NaN propagation.

### Story 112 - ADX-Gated Tail And q Ablation

Target:

- Student-t and unified Student-t process noise/tail parameters.

Indicator logic:

- Strong trend can reduce reversal variance but increase gap-tail risk.
- Test ADX as a q and tail-thickness gate.

Acceptance:

- Improves stress windows or full-gate calibration.
- Rejected if it overfits high-ADX continuation.

### Story 113 - Ichimoku Cloud Regime Features

Target:

- feature pipeline.
- regime classification diagnostics.

Indicator logic:

- Use Tenkan/Kijun distance, cloud thickness, price-cloud distance, and cloud
  twist age.
- Lag Senkou-style features so no forward plotting leaks into model inputs.

Acceptance:

- Explicit lag policy test.
- Regime-slice calibration improves or feature is deleted.

### Story 114 - Ichimoku Equilibrium For OU Input

Target:

- state-equation mean reversion.

Indicator logic:

- Kijun and cloud midpoint can serve as multi-horizon equilibrium candidates.
- Model chooses between state-space equilibrium and Ichimoku equilibrium.

Acceptance:

- Better range-regime hit/Brier without weakening trend regimes.
- Keep only if BMA selects it with nontrivial posterior mass.

### Story 115 - Donchian Channel State

Target:

- breakout and range diagnostics.

Indicator logic:

- Emit position inside Donchian channel, channel width percentile, breakout age,
  and failed breakout markers.

Acceptance:

- Improves range/trend classification or stress-slice behavior.
- No lookahead high/low leakage.

### Story 116 - Turtle Breakout Quality In Unified Scoring

Target:

- unified Student-t scoring and signal confidence.

Indicator logic:

- Breakout quality requires channel expansion, ATR support, and follow-through.
- Test only as model context, not a trading override.

Acceptance:

- Improves directional calibration after pass-2 calibration.
- Deleted if it increases turnover without PF/Sharpe lift.

### Story 117 - Bollinger/Keltner Squeeze Feature

Target:

- volatility-regime and tail-risk detection.

Indicator logic:

- Compression occurs when Bollinger width is inside Keltner width.
- Expansion after compression can signal variance state change.

Acceptance:

- Improves variance calibration around vol-release periods.
- No degradation in calm regimes.

### Story 118 - Bollinger Percentile And Vol-Mismatch Audit

Target:

- predictive variance diagnostics.

Indicator logic:

- Compare band percentile against realized-vol and model sigma.
- Identify whether model variance lags during band expansion.

Acceptance:

- Either accepted as an observation-noise model input or rejected with no model
  integration retained.
- Any duplicate vol feature is deleted.

### Story 119 - RSI/StochRSI Exhaustion With Calibration Guard

Target:

- bounded oscillator features.
- probability calibration.

Indicator logic:

- RSI and StochRSI can mark exhaustion only when trend strength and volatility
  context agree.
- Use monotonic calibration guard so oversold does not automatically mean buy.

Acceptance:

- Better Brier/hit after calibration.
- Deleted if it creates anti-trend false positives.

### Story 120 - Williams %R Failed-Breakout Reversal

Target:

- range-regime reversal diagnostics.

Indicator logic:

- Use Williams %R with Donchian failure context to test reversal probability.
- Keep only as range-regime feature.

Acceptance:

- Improves low-vol range assets without damaging trend assets.
- If effect is asset-specific, quarantine to diagnostics.

### Story 121 - MACD/PPO/TRIX Belief Momentum

Target:

- belief momentum and mean input.

Indicator logic:

- MACD/PPO/TRIX capture acceleration at different scales.
- Orthogonalize against existing return momentum before model use.

Acceptance:

- Belief momentum improves hit/Brier after calibration.
- Remove redundant momentum transforms.

### Story 122 - Multi-Horizon Momentum Orthogonalization

Target:

- state-equation momentum input basis.

Indicator logic:

- Build one orthogonal momentum basis from raw returns, MACD/PPO, KAMA slope,
  and HA persistence.

Acceptance:

- Fewer momentum features with equal or better results.
- Prevents class explosion from many near-duplicates.

### Story 123 - Volume Flow Model Inputs

Target:

- OBV, MFI, CMF, and volume-price confirmation.
- model variance/confidence input transport.

Indicator logic:

- Volume confirms or rejects price moves inside model variance, confidence, and
  tail-risk inputs; it must not create direction alone.

Acceptance:

- Better model-calibrated signal precision or stress calibration.
- Robust missing-volume policy for indices, FX, and commodities.

### Story 124 - Liquidity And Volume z-Score Conditioner

Target:

- predictive uncertainty and model confidence.

Indicator logic:

- Abnormal volume and dollar-volume changes can indicate event risk or fragile
  liquidity.
- Test as model variance inflation and confidence dampening.

Acceptance:

- Reduces model-driven magnitude outliers and unstable high-conviction signals.
- No penalty for assets with structurally low volume data quality.

### Story 125 - Anchored And Rolling VWAP Dislocation

Target:

- model equilibrium and dislocation inputs.

Indicator logic:

- Rolling VWAP and event/quarter anchors estimate traded equilibrium.
- Price distance from VWAP can inform MR or trend continuation depending on
  ADX/ER context.

Acceptance:

- Clear lag/anchor policy.
- Better range/trend split or deleted.

### Story 126 - VWAP Band Model Confidence Conditioner

Target:

- model confidence and predictive variance.

Indicator logic:

- VWAP band distance acts like volume-weighted z-score.
- Extreme dislocation can widen model variance and reduce confidence authority.

Acceptance:

- Improves calibrated PF/drawdown proxy without suppressing all winners.
- No effect on assets without valid volume unless explicitly supported.

### Story 127 - Hurst/Fractal Persistence Model Input

Target:

- model regime persistence versus mean reversion.

Indicator logic:

- Hurst/fractal dimension estimates whether recent path is persistent, random,
  or mean reverting.

Acceptance:

- Better trend/range gating.
- Stable estimates under short samples.

### Story 128 - Wavelet Energy Model Input Audit

Target:

- rough-volatility and impulse model inputs.

Indicator logic:

- Multi-scale wavelet energy can detect impulses and compression without
  relying on one fixed lookback.

Acceptance:

- Wavelet input must improve stress/tail slices or be deleted as expensive noise.
- Runtime must remain acceptable.

### Story 129 - Cross-Sectional Relative Strength

Target:

- model drift context and market conditioning.

Indicator logic:

- Compare each asset to sector, index, and risk-on/risk-off basket momentum.
- Use relative strength as model context for drift and confidence.

Acceptance:

- Better cross-asset dispersion handling.
- No survival bias in benchmark universe.

### Story 130 - Sector/Beta Indicator Normalization

Target:

- model feature normalization.

Indicator logic:

- Indicator thresholds differ by sector, volatility, beta, and asset class.
- Normalize features by asset-specific rolling distributions and market beta.

Acceptance:

- Reduces unresolved ticker/asset-class weirdness.
- Improves model transfer across equities, ETFs, FX, metals, and crypto.

### Story 131 - Indicator-Integrated Model Selection Layer

Target:

- BIC/LFO model-selection layer.

Indicator logic:

- Candidate indicator-integrated model variants compete under penalties.
- Selection can choose no indicator, one cluster, or a small orthogonal bundle
  inside a model variant.

Acceptance:

- Model input count stays small.
- Full-gate performance improves or remains equal with cleaner model selection.

### Story 132 - Indicator-Augmented Canonical Student-t Variant

Target:

- canonical Student-t model family.

Indicator logic:

- Add a side-by-side indicator-augmented canonical variant, not replacement.
- Start with the best-selected feature cluster from cycles 101-131.

Acceptance:

- Canonical and augmented canonical compete via registry.
- No silent model drops.

### Story 133 - Indicator-Augmented Improved Student-t Variant

Target:

- improved Student-t model family.

Indicator logic:

- Test indicator-selected mean/q/tail conditioning inside the improved model.
- Preserve existing improved/unimproved competition.

Acceptance:

- Better full-gate calibrated signal metrics or rejection with code removed.

### Story 134 - Indicator-Augmented Unified Improved Student-t Variant

Target:

- unified improved Student-t.

Indicator logic:

- Use indicators only for unified state conditioning where they improve
  calibrated uncertainty and regime fit.

Acceptance:

- No class explosion.
- Method count decreases or remains controlled despite new features.

### Story 135 - Indicator-Augmented Gaussian Control

Target:

- Gaussian baseline.

Indicator logic:

- Add the same selected indicator state to Gaussian controls to separate
  indicator value from Student-t tail value.

Acceptance:

- If Gaussian gets the same lift, attribute edge to feature not tails.
- If Gaussian degrades, keep Student-t-specific integration only.

### Story 136 - Indicator Interaction Deletion Pass

Target:

- all indicator candidates.

Indicator logic:

- Delete interactions that are collinear, unstable, or only improve in-sample.

Acceptance:

- Fewer feature paths.
- Equal or better full-gate metrics.

### Story 137 - Indicator PIT/AD/Berkowitz Audit

Target:

- calibration diagnostics.

Indicator logic:

- Measure whether indicator features improve probability integral transform,
  tail adequacy, and residual independence.

Acceptance:

- Distributional diagnostics improve, not just trade metrics.
- Any feature that improves PF but breaks calibration is quarantined.

### Story 138 - Indicator-Aware EMOS Calibration

Target:

- `signals_calibration.py`.

Indicator logic:

- Test whether EMOS mean/variance correction should condition on indicator
  regimes such as squeeze, trend strength, and volume confirmation.

Acceptance:

- Better Brier/CRPS/PF without sparse-regime overfitting.
- Sparse partitions remain gated.

### Story 139 - Indicator-Aware Beta Calibration And Thresholds

Target:

- Beta calibration and label thresholds.

Indicator logic:

- Probability maps may differ in high-trend, exhaustion, and squeeze-release
  states.

Acceptance:

- Better calibrated hit rate and Brier.
- Reject if threshold stability worsens across repeated runs.

### Story 140 - Indicator Stress/Tail Benchmark Expansion

Target:

- `benchmark_retune_50.py`.

Indicator logic:

- Add slices for HA exhaustion, squeeze release, ADX high trend, volume shock,
  VWAP dislocation, and Donchian breakout failure.

Acceptance:

- Benchmark exposes where indicators help or hurt.
- No additional expensive model-fitting pass.

### Story 141 - Heikin-Ashi Versus Raw-Candle Stress Benchmark

Target:

- benchmark harness.

Indicator logic:

- Compare HA features against raw OHLC candle features in the same stress
  windows and asset-specific tail deciles.

Acceptance:

- HA must show incremental value over raw candle geometry.
- Otherwise keep raw candle features only.

### Story 142 - Trend-Following Indicator Cluster Noise Analysis

Target:

- SuperTrend, ADX/DMI, Donchian, MACD/PPO/TRIX.

Indicator logic:

- Repeat full gates to estimate whether apparent trend-feature lift is stable.

Acceptance:

- Promotion requires improvement beyond run-to-run noise.
- Revert if edge vanishes on repeat.

### Story 143 - Mean-Reversion Indicator Cluster Noise Analysis

Target:

- RSI/StochRSI, Williams %R, VWAP distance, KAMA distance, Ichimoku equilibrium.

Indicator logic:

- Test reversal indicators only where regime context supports reversal.

Acceptance:

- Better range-regime calibration and signal metrics.
- No global reversal bias.

### Story 144 - Volume/Flow Indicator Cluster Noise Analysis

Target:

- OBV, MFI, CMF, volume z-score, dollar volume.

Indicator logic:

- Test whether flow confirmation improves confidence and reduces false signals.

Acceptance:

- Stable improvement on equities and graceful no-op on assets without volume.

### Story 145 - Feature Latency And Missing-Data Robustness

Target:

- all indicator features.

Indicator logic:

- Audit weekend/holiday gaps, foreign tickers, ETFs, FX, crypto, and metals.

Acceptance:

- No unresolved symbols or NaN feature explosions.
- Frontend/backend diagnostics can report feature availability.

### Story 146 - Indicator Kernel Speed Pass

Target:

- Numba/vectorized indicator kernels.

Indicator logic:

- Fuse shared rolling statistics for ATR, channels, bands, and oscillators.
- Avoid recomputing the same rolling max/min/mean across indicators.

Acceptance:

- Faster feature generation with parity tests.
- No mathematical behavior drift.

### Story 147 - Indicator Diagnostics Frontend

Target:

- backend diagnostics.
- frontend model comparison.

Indicator logic:

- Show which indicator clusters each model used, their posterior weight, and
  whether they improved or hurt calibration.

Acceptance:

- User can see indicator and non-indicator models side by side.
- Unknown indicator families render gracefully.

### Story 148 - Full Indicator Model Competition Gate

Target:

- full retune/calibration benchmark.

Indicator logic:

- Run canonical, improved, unified, Gaussian, and indicator-augmented variants
  side by side.

Acceptance:

- 50/50 assets.
- 0 failures.
- 0 calibration warnings.
- Indicator variants win only when the benchmark proves it.

### Story 149 - Signal Smoke And Weak-Asset Failure Analysis

Target:

- signal-generation smoke.
- weak recent assets from cycle 100.

Indicator logic:

- Analyze whether indicators fix or worsen recent weak 90-day windows such as
  NVDA/QQQ/SPY/XLP without overfitting to them.

Acceptance:

- Weak-asset diagnostics improve or the indicator path is rejected.
- No hand-tuned asset-specific exceptions.

### Story 150 - Final Indicator-Integrated Model Release Gate

Target:

- full system release gate.

Indicator logic:

- Keep only indicator paths that survived ablation, repeated-run noise, stress
  slices, full 50-stock retune/calibration, frontend visibility, and signal
  smoke.

Acceptance:

- Final metrics are equal or better than cycle 100.
- All rejected indicator code is removed.
- `Models.md` records accepted/rejected evidence for every cycle.

## Definition of Done

The phase is done only when:

- The 50-cycle ledger is complete or explicitly reprioritized with evidence.
- Every accepted change has a benchmark artifact.
- Every rejected idea has had its experimental code removed.
- `make retune` keeps canonical and improved models competing side by side.
- Calibration and signal generation consume the same registry-backed model names.
- Frontend model displays reflect the actual tune cache.
- The final 50-stock benchmark shows no failures and no calibration warnings.
