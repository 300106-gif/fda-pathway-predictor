# Model Evaluation Report — FDA Pathway Predictor
**Best Model:** `RandomForest`
**Best Macro F1:** `1.0000`

---

## RandomForest ✓ (selected)
**Macro F1:** `1.0000`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       1.00      1.00      1.00       412
         PMA       1.00      1.00      1.00       188

    accuracy                           1.00       600
   macro avg       1.00      1.00      1.00       600
weighted avg       1.00      1.00      1.00       600

```

## GradientBoosting
**Macro F1:** `1.0000`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       1.00      1.00      1.00       412
         PMA       1.00      1.00      1.00       188

    accuracy                           1.00       600
   macro avg       1.00      1.00      1.00       600
weighted avg       1.00      1.00      1.00       600

```

## LogisticRegression
**Macro F1:** `0.3333`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.00      0.00      0.00       412
     De Novo       0.00      0.00      0.00         0
         PMA       1.00      1.00      1.00       188

    accuracy                           0.31       600
   macro avg       0.33      0.33      0.33       600
weighted avg       0.31      0.31      0.31       600

```
