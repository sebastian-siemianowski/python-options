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

## Definition of Done

The phase is done only when:

- The 50-cycle ledger is complete or explicitly reprioritized with evidence.
- Every accepted change has a benchmark artifact.
- Every rejected idea has had its experimental code removed.
- `make retune` keeps canonical and improved models competing side by side.
- Calibration and signal generation consume the same registry-backed model names.
- Frontend model displays reflect the actual tune cache.
- The final 50-stock benchmark shows no failures and no calibration warnings.
