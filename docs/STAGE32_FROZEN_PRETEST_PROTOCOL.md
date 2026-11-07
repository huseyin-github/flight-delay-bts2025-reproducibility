# Stage32 frozen protocol — before February outcome evaluation

Status: Frozen, pre-test. No February arrival-delay model scores were examined before this commit.

## Data
- January training: 2025-01-01 to 2025-01-31, 578957 eligible flights; workbook `TranstatsData_Ayse5.xlsx`, length 85610376 bytes.
- February independent test: 2025-02-01 to 2025-02-28, raw `BTS_2025_02_Raw.csv`, 559577 original rows, 228203925 bytes, SHA-256 `a6fd508e7e15471f5bbcb0f92d66e8a2381a0bac1eb8d49776c0ecad7a91ecb6`.
- The original February CSV will remain unmodified.

## Frozen methods
- **Operational prediction instant:** after actual gate departure, once observed DEP_DELAY becomes available.
- **Target:** ARR_DELAY minutes; no transformation or 15-minute filtering.
- **Inputs (12):** DAY_OF_MONTH, DAY_OF_WEEK, MKT_CARRIER, ORIGIN_AIRPORT, DEST_AIRPORT, CRS_ELAPSED_TIME, DISTANCE, DEP_DELAY, and sine/cosine encodings of CRS_DEP_TIME and CRS_ARR_TIME.
- **Eligibility:** not cancelled/diverted; ARR_DELAY and DEP_DELAY available; complete selected predictors.
- **Source mapping:** February ORIGIN and DEST map to January ORIGIN_AIRPORT and DEST_AIRPORT. January-only category vocabularies and modal fallback are reused.
- **Principal frozen January-fitted models:** native MATLAB fitlm RobustOpts=off (Linear); native MATLAB fitlm RobustOpts=on (RobustLinear); native MATLAB fitrsvm linear kernel KernelScale=auto Standardize=true CacheSize=1000MB (LinearSVM). All use 578957 January observations. No refitting on February.
- The prior six-pass sparse-QR/IRLS numerical audit is NOT the native MATLAB Robust Linear model.
- Previously explored residual LSBoost is not one of the 23 native MATLAB presets and is EXCLUDED from confirmatory model comparison. Its separate exploratory status must be disclosed in the revised manuscript.

## Frozen analyses
- Report R2, RMSE (minutes), and MAE (minutes) for January training, earlier January expanding-window temporal cross-validation, and February independent testing (labeling distinct model fits).
- Baselines: January training mean, January training median, and direct DEP_DELAY.
- Robust Linear actual-versus-predicted scatter with 1:1 line, residuals versus predictions, residuals versus actuals.
- Severe-delay diagnostics: observed ARR_DELAY >=60 minutes and >=120 minutes.
- Paired RMSE and MAE differences: 2000 bootstrap draws of 28 calendar-day clusters, seed 42032, 95% percentile intervals.

## Evaluator integrity
MATLAB evaluator: `Stage32_February_Independent_Test.m`
SHA-256: `d327ac58e3bc8d1a1334c11fd4e2a2f20ee8daa2d2fee6e94b2dc3604bbaefde`.

Once February results are observed, no model selection, tuning or evaluation protocol changes may be presented as untouched confirmation. Code failures before scores can only be corrected transparently without adapting scientific choices to test results.