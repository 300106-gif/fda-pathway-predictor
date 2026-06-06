# EDA Insights — FDA Pathway Predictor

## Dataset Overview
- **Total records**: 2,999
- **Feature columns**: `device_name`, `product_code`, `device_class`, `medical_specialty_description`, `decision_code`, `decision_date`, `implant_flag`, `life_sustain_support_flag`, `k_number`, `source`, `decision_year`
- **Target**: `pathway`
  - 510k: 1,999 (66.7%), PMA: 999 (33.3%), De Novo: 1 (0.0%)

## Key Findings

### 1. Class Imbalance
`510k` dominates at **66.7%** of records.
Use stratified splitting and **macro F1** as the primary evaluation metric.

### 2. Device Class — Strong Signal
Class III devices are almost exclusively in the **PMA** pathway.
Class II maps predominantly to **510(k)**. This feature is a strong predictor.

### 3. Top Medical Specialty
The most frequent specialty is **Unknown**, reflecting openFDA dataset composition.
Medical specialty is expected to be a useful categorical feature.

### 4. Implant Flag
**0.0%** of PMA records are implants, confirming this binary flag
is a meaningful predictor for the PMA vs. 510(k) distinction.

### 5. Modeling Recommendations
- Apply **stratified train/test split** (80/20) to preserve class ratios.
- Label-encode `device_class` and `medical_specialty_description`.
- Binarize `implant_flag` and `life_sustain_support_flag` as 0/1 integers.
- Report **per-class precision, recall, F1** in addition to overall accuracy.
- Consider class-weight adjustments if De Novo recall is insufficient.
