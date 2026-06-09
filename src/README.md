# STAI-X Challenge 2026 - Expert Jasmine

## Results

### Out-of-Fold Validation Performance

| Category | MAE | 
|----------|---|
| all_drugs | 0.3705 |
| all_opioids | 0.1725 |
| all_stimulants | 0.1023 |
| **Block Average** | **0.2151** |

### Model Configuration

- **Features**: 79 total (57 base + 22 target lags)
- **Algorithm**: LightGBM with MAE optimization
- **Architecture**: Category-specific models (3 models)
- **Validation**: 3-fold time-series cross-validation

### Submission Statistics

- **Total Predictions**: 918
- **Mean Rate**: 15.19 per 10,000 ED visits
- **Range**: 3.49 - 57.22
- **Validation**: ✓ All checks passed
