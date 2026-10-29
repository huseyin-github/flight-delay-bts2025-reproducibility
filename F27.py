"""Validate CSV summaries transcribed from an author-provided MATLAB console log.
No third-party packages or access to the underlying dataset is required.
This checks structural consistency ONLY; it does not independently rerun ML.
"""
import csv
from pathlib import Path
root = Path(__file__).resolve().parents[1]
with (root / 'results' / 'Stage27_FoldMetrics_transcribed.csv').open(encoding='utf-8', newline='') as f:
    folds = list(csv.DictReader(f))
with (root / 'results' / 'Stage27_PooledOOF_transcribed.csv').open(encoding='utf-8', newline='') as f:
    pooled = list(csv.DictReader(f))
assert len(folds) == 115, f'Expected 115 fold/preset records, got {len(folds)}'
assert len(pooled) == 23, f'Expected 23 models, got {len(pooled)}'
assert len({(r['Fold'],r['Preset']) for r in folds}) == 115
assert sum(r['Status'] == 'OK' for r in folds) == 82
assert sum(r['Status'] == 'COMPLETE_5_OF_5' for r in pooled) == 16
for row in pooled:
    k = sum(r['Status'] == 'OK' for r in folds if r['Preset'] == row['Preset'])
    assert int(row['CompletedFolds']) == k, (row['Preset'], k)
    if k < 5:
        assert row['RMSE_min'].lower() == 'nan', row
best = min((r for r in pooled if int(r['CompletedFolds']) == 5), key=lambda r: float(r['RMSE_min']))
assert best['Preset'] == 'Robust Linear', best
assert abs(float(best['RMSE_min']) - 15.259) < 1e-6
print('Structural checks passed: 23 presets, 115 slots, 82 successful fits, 16 fully evaluated presets.')
print('Lowest pooled RMSE among complete presets:', best['Preset'], best['RMSE_min'], 'minutes')
print('Note: source data, fitting and test results have NOT been independently rerun.')
