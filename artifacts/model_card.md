# Model Card — FDA Regulatory Pathway Predictor

## Model Details
- **Name**: GradientBoosting
- **Version**: 1.0.0
- **Framework**: scikit-learn
- **Task**: Multi-class classification (3 classes)
- **Classes**: `510k`, `De Novo`, `PMA`

## Intended Use
Predict the most likely FDA regulatory pathway for a medical device based on its
characteristics. **For planning purposes only — not a substitute for regulatory counsel.**

## Training Data
| Property | Value |
|----------|-------|
| Source | openFDA API (510k + PMA endpoints) or synthetic fallback |
| Split | 80% train / 20% test, stratified on `pathway` |
| Random seed | 42 |
| Class weighting | balanced (where supported) |

## Input Features
| Feature | Encoding | Description |
|---------|----------|-------------|
| `device_class_enc` | Label-encoded int | Device risk classification (Class I/II/III) |
| `medical_specialty_description_enc` | Label-encoded int | Primary medical specialty |
| `implant_flag_bin` | Binary (0/1) | Whether the device is implantable |
| `life_sustain_support_flag_bin` | Binary (0/1) | Whether the device is life-sustaining |

## Performance Summary
- **Best Model**: GradientBoosting
- **Macro F1**: `0.7159`

See `evaluation_report.md` for full per-class precision, recall, and confusion matrix.

## Limitations
- Trained on publicly available FDA data; actual pathway decisions depend on many
  additional factors (predicate device history, clinical evidence, etc.).
- De Novo pathway is under-represented; recall for that class may be lower.
- Model does not account for recent regulatory guidances or policy changes.

## Ethical Considerations
This model is strictly a **planning aid** for early-stage regulatory strategy.
Final decisions must involve qualified regulatory affairs professionals.

## Disclaimer
> This tool is for planning purposes only. Consult qualified regulatory counsel.
