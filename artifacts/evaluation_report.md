# Model Evaluation Report — FDA Pathway Predictor
**Best Model:** `GradientBoosting`
**Best Macro F1:** `0.7159`

---

## RandomForest
**Macro F1:** `0.6712`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.88      0.90      0.89       260
     De Novo       0.34      0.68      0.45        95
         PMA       0.83      0.57      0.67       340

    accuracy                           0.71       695
   macro avg       0.68      0.72      0.67       695
weighted avg       0.78      0.71      0.72       695

```

## GradientBoosting ✓ (selected)
**Macro F1:** `0.7159`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.88      0.90      0.89       260
     De Novo       0.53      0.39      0.45        95
         PMA       0.79      0.83      0.81       340

    accuracy                           0.80       695
   macro avg       0.73      0.71      0.72       695
weighted avg       0.79      0.80      0.79       695

```

## LogisticRegression
**Macro F1:** `0.3812`

**Full Classification Report:**
```
              precision    recall  f1-score   support

        510k       0.56      0.73      0.63       260
     De Novo       0.22      0.73      0.34        95
         PMA       0.70      0.10      0.17       340

    accuracy                           0.42       695
   macro avg       0.49      0.52      0.38       695
weighted avg       0.58      0.42      0.37       695

```
