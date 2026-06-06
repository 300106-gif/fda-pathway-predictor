# EDA Insights — FDA Pathway Predictor

## Dataset Overview
- **Total records**: 200
- **Feature columns**: `device_name`, `product_code`, `device_class`, `medical_specialty_description`, `implant_flag`, `life_sustain_support_flag`
- **Target**: `pathway`
  - 510k: 123 (61.5%), PMA: 61 (30.5%), De Novo: 16 (8.0%)

## Key Findings

### 1. Class Imbalance
`510k` dominates at **61.5%** of records.
Use stratified splitting and **macro F1** as the primary evaluation metric.

### 2. Device Class — Strong Signal
Class III devices are almost exclusively in the **PMA** pathway.
Class II maps predominantly to **510(k)**. This feature is a strong predictor.

### 3. Top Medical Specialty
The most frequent specialty is **General Hospital**, reflecting openFDA dataset composition.
Medical specialty is expected to be a useful categorical feature.

### 4. Implant Flag
**62.3%** of PMA records are implants, confirming this binary flag
is a meaningful predictor for the PMA vs. 510(k) distinction.

### 5. Modeling Recommendations
- Apply **stratified train/test split** (80/20) to preserve class ratios.
- Label-encode `device_class` and `medical_specialty_description`.
- Binarize `implant_flag` and `life_sustain_support_flag` as 0/1 integers.
- Report **per-class precision, recall, F1** in addition to overall accuracy.
- Consider class-weight adjustments if De Novo recall is insufficient.
