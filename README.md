# 🧠 Classification of Twelve Types of Depression
**Tool:** Orange Data Mining  
**Dataset:** Mental Health Classification (1,998 instances, 21 features)  
**Task:** Multi-class depression classification (12 classes)  
**Models compared:** Decision Tree, Random Forest, Logistic Regression, Naive Bayes

---

## 📁 Dataset Overview

The dataset is derived from a structured mental health and depression survey containing 1,998 cleaned responses. It includes 21 demographic, lifestyle, behavioral, and psychological features.

**Target Variable:** `Depression_Type` — numerically encoded across 12 classes:

| Code | Depression Type |
|------|----------------|
| 0 | No clinically significant depression |
| 1 | Minimal / Mild depression |
| 2 | Moderate depression |
| 3 | Moderately-severe depression |
| 4 | Severe depression |
| 5 | Persistent depressive disorder (Dysthymia) |
| 6 | Seasonal affective pattern |
| 7 | Peripartum / Postpartum depression |
| 8 | Bipolar-related depressive episode |
| 9 | Situational / Reactive depression |
| 10 | Psychotic depression |
| 11 | Other specified depressive disorder |

**Class distribution is imbalanced** — e.g., Type 9 (Situational) has 627 samples while Type 8 (Bipolar) has only 21 samples.

---

## ⚠️ Data Leakage Investigation

### Problem: Models Initially Scored Perfect 1.000

When the dataset was first loaded and all 21 features were included, Decision Tree and Random Forest both scored **1.000 across all metrics** (AUC, CA, F1, Precision, Recall, MCC). This is a red flag — real-world mental health data never achieves perfect classification.

### Root Cause: Synthetic Data Encoding Artifacts

This dataset was **synthetically generated**, meaning the `Depression_Type` label was used to construct certain feature values during data creation — not the other way around. This creates a circular relationship where some feature values perfectly and exclusively map to one depression type, making classification trivially easy and meaningless.

### Leaky Features Identified

By analyzing the cross-tabulation of each feature against `Depression_Type`, the following features were found to contain values that **exclusively appear in one depression class**:

| Feature | Leaky Value(s) | Always maps to Depression_Type |
|---------|---------------|-------------------------------|
| `Symptoms` | 2 | 9 (Situational) |
| `Symptoms` | 4 | 10 (Psychotic) |
| `Symptoms` | 6 | 4 (Severe) |
| `Symptoms` | 9 | 5 (Dysthymia) |
| `Symptoms` | 10 | 7 (Peripartum) |
| `Symptoms` | 12 | 1 (Mild) |
| `Symptoms` | 14 | 11 (Other) |
| `Your overeating level` | 1, 2, 3, 6, 12 | Types 4, 6, 5, 6, 0 respectively |
| `Coping_Methods` | 1, 8, 12 | Types 5, 2, 8 respectively |
| `Low_Energy` | 2 | 6 (Seasonal) |
| `Low_SelfEsteem` | 2 | 2 (Moderate) |
| `Education_Level` | 3 | 2 (Moderate) |

Together, these leaky values cover **all 12 depression classes**, which is why tree-based models could achieve perfect scores — they simply learn these lookup rules rather than any meaningful pattern.

### Proof of Leakage (Orange Workflow)

To verify each leaky value visually in Orange:
1. Add a **Select Rows** widget with a single condition (e.g., `Symptoms = 2`)
2. Connect to a **Distributions** widget
3. Set the variable to `Depression_Type`
4. Result: a **single bar** appears at exactly one class = confirmed leakage

Each leaky value was verified this way. The status bar in Orange shows `38 | 1` format — meaning N rows all belonging to **1 unique class**.

### Features Removed (Skipped in Orange)

The following columns were set to **Skip** in the Orange File widget to eliminate leakage:

- `Symptoms`
- `Your overeating level`
- `Coping_Methods`
- `Low_Energy`
- `Low_SelfEsteem`
- `Education_Level`

> **Note:** `Depression_Score` was investigated and confirmed **clean** — each score value maps to multiple depression types with near-zero correlation (-0.02) to the target, so it was kept.

---

## 🔬 Experimental Setup

**Tool:** Orange Data Mining (visual ML workflow)  
**Evaluation method:** 5-Fold Cross Validation  
**Why 5-fold CV?** The data is split 5 times; each time a different 20% is used as the test set, and results are averaged. This is more reliable than a single train/test split and is the standard approach for datasets of this size.

**Models and their default settings in Orange:**

| Model | Key Settings |
|-------|-------------|
| Decision Tree | Max depth: 100, Min instances in leaves: 2 |
| Random Forest | 10 trees, Do not split subsets smaller than 5 |
| Logistic Regression | Lasso (L1) regularization, C=1 |
| Naive Bayes | Default settings |

---

## 📊 Evaluation Metrics

| Metric | Full Name | What it means |
|--------|-----------|---------------|
| AUC | Area Under ROC Curve | How well the model separates classes. 1.0 = perfect, 0.5 = random guessing |
| CA | Classification Accuracy | % of predictions that were correct overall |
| F1 | F1 Score | Balance between Precision and Recall — best for imbalanced classes |
| Prec | Precision | Of all cases predicted as X, how many were actually X |
| Recall | Recall | Of all actual X cases, how many did it correctly find |
| MCC | Matthews Correlation Coefficient | Most honest single score — accounts for class imbalance well |

> **For this project, F1 and MCC are the most important metrics** since we have 12 unbalanced classes. Accuracy (CA) alone can be misleading when class sizes differ greatly.

---

## 📈 Results

### Before Removing Leaky Features (Inflated / Invalid)

| Model | AUC | CA | F1 | Prec | Recall | MCC |
|-------|-----|----|----|------|--------|-----|
| Tree | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.933 | 0.744 | 0.739 | 0.766 | 0.744 | 0.684 |
| Logistic Regression | 0.914 | 0.723 | 0.715 | 0.716 | 0.723 | 0.654 |

⚠️ Perfect scores for Tree and Random Forest indicate data leakage — not genuine model performance.

### After Removing Leaky Features (Valid Results)

| Model | AUC | CA | F1 | Prec | Recall | MCC |
|-------|-----|----|----|------|--------|-----|
| Tree | 0.993 | 0.985 | 0.985 | 0.986 | 0.985 | 0.981 |
| Random Forest | 0.999 | 0.961 | 0.962 | 0.962 | 0.961 | 0.952 |
| Logistic Regression | 0.836 | 0.577 | 0.556 | 0.568 | 0.577 | 0.463 |
| Naive Bayes | 0.852 | 0.583 | 0.577 | 0.614 | 0.583 | 0.484 |

> **Note:** Tree and Random Forest still score very high (~0.985 and ~0.961 CA) even after leakage removal. This is likely due to the synthetic nature of the dataset — the remaining features still follow very structured patterns not typical of real survey data. These results should be interpreted with that caveat.

---

## 🏆 Model Comparison Summary

**Best overall model: Decision Tree** (CA: 0.985, F1: 0.985, MCC: 0.981)  
**Runner-up: Random Forest** (CA: 0.961, F1: 0.962, MCC: 0.952) — slightly lower but more robust to overfitting in general  
**Weakest models: Logistic Regression and Naive Bayes** — both around 57–58% accuracy, suggesting the decision boundary for this classification task is non-linear

The statistical comparison table in Orange (probability that one model scores higher than another) confirmed that Tree and Random Forest are significantly better than both Logistic Regression and Naive Bayes (p ≈ 1.000), while Tree and Random Forest are not significantly different from each other (p = 0.004).

---

## 📝 Key Takeaways

1. **Always check for data leakage** before trusting perfect or near-perfect scores — especially with synthetically generated datasets
2. **Tree-based models dominate** on structured/synthetic data; linear models (LR, NB) struggle with non-linear class boundaries
3. **F1 and MCC are better indicators** than raw accuracy for imbalanced multi-class problems
4. **5-fold cross-validation** is the appropriate evaluation strategy for this dataset size
5. **The dataset has synthetic origins** — results may not generalize to real clinical data; this should be acknowledged in any academic use

---

## 🗂️ Features Used in Final Model (15 of 21)

`Gender`, `Age`, `Employment_Status`, `Low_SelfEsteem` *(clean values only)*, `Search_Depression_Online`, `Worsening_Depression`, `How many times you eat`, `SocialMedia_Hours`, `SocialMedia_WhileEating`, `Sleep_Hours`, `Nervous_Level`, `Depression_Score`, `Self_Harm`, `Mental_Health_Support`, `Suicide_Attempts`

**Skipped (leaky):** `Symptoms`, `Your overeating level`, `Coping_Methods`, `Low_Energy`, `Low_SelfEsteem` *(leaky value)*, `Education_Level` *(leaky value)*
