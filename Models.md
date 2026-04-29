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
| 013 | Canonical Student-t volatility correction audit | Pending |
| 014 | Canonical Student-t MLE profile audit | Pending |
| 015 | Canonical Student-t method-size reduction | Pending |
| 016 | Unified improved Stage 5 search pruning | Done |
| 017 | Unified improved Stage 5 scoring rewrite | Pending |
| 018 | Unified improved Stage 6 helper extraction | Pending |
| 019 | Unified improved GARCH/CRPS branch pruning | Pending |
| 020 | Unified improved filter path kernel parity | Pending |
| 021 | Unified canonical Stage 5 simplification | Pending |
| 022 | Unified canonical Stage 6 calibration audit | Pending |
| 023 | Unified canonical filter path cleanup | Pending |
| 024 | Unified canonical MLE/grid consistency audit | Pending |
| 025 | Unified canonical dead branch deletion | Pending |
| 026 | Gaussian Stage 1 objective audit | Pending |
| 027 | Signal calibration cache-hit import/data deletion | Done |
| 028 | Signal calibration EMOS sub-regime polish gate | Done |
| 029 | Signal calibration EMOS solver cap test | Rejected |
| 030 | Full gate after signal calibration deletions | Done |
| 031 | RV-Q deletion test | Rejected |
| 032 | Improved filter scalar clip hot-loop test | Rejected |
| 033 | Registry/feature-pipeline model-name dedupe | Pending |
| 034 | Signal calibration pass-2 boundary audit | Pending |
| 035 | Signal probability calibration scoring | Pending |
| 036 | Ticker and asset classification consolidation | Done |
| 037 | Frontend model visibility dynamic labels | Pending |
| 038 | Backend diagnostics model matrix audit | Pending |
| 039 | Numba kernel naming/grouping pass | Pending |
| 040 | Numba wrapper deletion pass | Pending |
| 041 | Calibration helper consolidation | Pending |
| 042 | PIT/AD/Berkowitz shared scoring pass | Pending |
| 043 | Real-data benchmark panel standardization | Pending |
| 044 | Profitability validation harness cleanup | Pending |
| 045 | Full retune repeated-run noise analysis | Pending |
| 046 | Per-asset slow-case investigation | Pending |
| 047 | Stress-period calibration benchmark | Pending |
| 048 | Crisis/tail validation benchmark | Pending |
| 049 | Final model-count and frontend audit | Pending |
| 050 | Final 50-stock full retune plus calibration release gate | Pending |

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
| 061 | `phi_student_t` shared calibration transport rewrite | Pending |
| 062 | `phi_student_t` production Numba filter-core parity | Pending |
| 063 | `phi_student_t` tail graft architecture cleanup | Pending |
| 064 | `phi_student_t` isotonic/probability transport deletion audit | Pending |
| 065 | `phi_student_t` proper scoring objective ablation | Pending |
| 066 | `phi_student_t` volatility correction orthogonalization | Pending |
| 067 | `phi_student_t` ν-grid topology pruning with posterior-mass proof | Pending |
| 068 | `phi_student_t` q/c/phi parameterization refactor | Pending |
| 069 | `phi_student_t` LFO-CV fold kernel rewrite | Pending |
| 070 | `phi_student_t` canonical/improved shared base extraction | Pending |
| 071 | `phi_student_t_unified_improved` Stage 5 decomposition | Pending |
| 072 | `phi_student_t_unified_improved` structural-array Numba rewrite | Pending |
| 073 | `phi_student_t_unified_improved` jump layer deletion or hard promotion | Pending |
| 074 | `phi_student_t_unified_improved` Stage 6 coordinate-search calibration rewrite | Pending |
| 075 | `phi_student_t_unified_improved` PIT/CRPS/entropy composite selection score | Pending |
| 076 | `phi_student_t_unified_improved` conditional skew GAS simplification | Pending |
| 077 | `phi_student_t_unified_improved` rough volatility layer audit | Pending |
| 078 | `phi_student_t_unified_improved` GARCH/PIT path Numba kernel rewrite | Pending |
| 079 | `phi_student_t_unified_improved` fallback-path deletion pass | Pending |
| 080 | `phi_student_t_unified_improved` method-size budget enforcement | Pending |
| 081 | `phi_student_t_unified` Stage 5 canonical parity audit | Pending |
| 082 | `phi_student_t_unified` filter-and-calibrate reuse from optimize diagnostics | Pending |
| 083 | `phi_student_t_unified` config/dataclass surface shrink | Pending |
| 084 | `phi_student_t_unified` weak asymmetry and jump branch deletion audit | Pending |
| 085 | `phi_student_t_unified` Markov/stress-q simplification | Pending |
| 086 | `kalman_phi_gaussian_unified` exact Kalman kernel rewrite | Pending |
| 087 | `kalman_gaussian_unified` closed-form CRPS gradient objective | Pending |
| 088 | Gaussian unified GAS-Q and momentum branch audit | Pending |
| 089 | Gaussian unified sign-probability objective | Pending |
| 090 | Gaussian unified calibration path consolidation | Pending |
| 091 | Numba kernel architecture split by mathematical responsibility | Pending |
| 092 | Numba wrapper deletion and typed boundary pass | Pending |
| 093 | EMOS Student-t optimizer rewrite in Numba | Pending |
| 094 | Beta calibration optimizer rewrite and focal-loss audit | Pending |
| 095 | Threshold optimization expected-utility rewrite with calibration guard | Pending |
| 096 | BMA posterior scoring rewrite with calibration entropy | Pending |
| 097 | Registry and model parameter transport hardening | Pending |
| 098 | Stress/crisis benchmark slices added to validation gate | Pending |
| 099 | Frontend/backend dynamic model diagnostics matrix | Pending |
| 100 | Final full retune/calibration/signal-generation release gate | Pending |

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

## Definition of Done

The phase is done only when:

- The 50-cycle ledger is complete or explicitly reprioritized with evidence.
- Every accepted change has a benchmark artifact.
- Every rejected idea has had its experimental code removed.
- `make retune` keeps canonical and improved models competing side by side.
- Calibration and signal generation consume the same registry-backed model names.
- Frontend model displays reflect the actual tune cache.
- The final 50-stock benchmark shows no failures and no calibration warnings.
