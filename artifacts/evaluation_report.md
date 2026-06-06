# Model Evaluation Report — FDA Pathway Predictor
**Best Model:** `GradientBoosting`
**Best Macro F1:** `0.6313`

---

## RandomForest
**Macro F1:** `0.4407`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.00      0.00      0.00       400
     De Novo       0.19      1.00      0.32        95
         PMA       1.00      1.00      1.00       200

    accuracy                           0.42       695
   macro avg       0.40      0.67      0.44       695
weighted avg       0.31      0.42      0.33       695

```

## GradientBoosting ✓ (selected)
**Macro F1:** `0.6313`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.81      1.00      0.89       400
     De Novo       0.00      0.00      0.00        95
         PMA       1.00      1.00      1.00       200

    accuracy                           0.86       695
   macro avg       0.60      0.67      0.63       695
weighted avg       0.75      0.86      0.80       695

```

## LogisticRegression
**Macro F1:** `0.4407`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.00      0.00      0.00       400
     De Novo       0.19      1.00      0.32        95
         PMA       1.00      1.00      1.00       200

    accuracy                           0.42       695
   macro avg       0.40      0.67      0.44       695
weighted avg       0.31      0.42      0.33       695

```
