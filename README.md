# 🧠 Classification of Twelve Types of Depression

> Multi-class depression classification using machine learning on a structured mental health survey dataset.

**Live App:** [classification-of-twelve-types-of-d.vercel.app](https://classification-of-twelve-types-of-d.vercel.app)

---

## 📁 Repository Structure

```
Classification-of-Twelve-Types-of-Depression/
│
├── data/
│   └── Mental_Health_Classification.csv    # Raw dataset (1,998 instances, 21 features)
│
├── models/                  # Trained model files (.pkl) + training notebook
│   ├── models.ipynb                        # Full training pipeline (RF, LR, SVM)
│   ├── random_forest.pkl                   # Saved Random Forest model
│   ├── logistic_regression.pkl             # Saved Logistic Regression model
│   └── SVM.pkl                             # Saved SVM model
│
├── depression_app/                         # Web application source (deployed on Vercel)
│
├── classification.ows                      # Orange Data Mining workflow file
└── README.md
```


---

## 📊 Dataset Overview

- **Source:** Structured mental health and depression survey
- **Instances:** 1,998 cleaned responses
- **Features:** 21 (demographic, lifestyle, behavioral, psychological)
- **Task:** Multi-class classification — 12 depression types
- **Target Variable:** `Depression_Type`

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

> ⚠️ **Class imbalance present** — Type 9 has 627 samples while Type 8 has only 21 samples.

---

## ⚠️ Data Leakage Investigation

### Problem
When all 21 features were included, Decision Tree and Random Forest both scored a **perfect 1.000** across all metrics, which is an immediate red flag.

### Leaky Features Identified & Removed

| Feature | Leaky Value(s) | Always maps to |
|---------|---------------|----------------|
| `Symptoms` | 2, 4, 6, 9, 10, 12, 14 | 7 different classes |
| `Your overeating level` | 1, 2, 3, 6, 12 | 5 different classes |
| `Coping_Methods` | 1, 8, 12 | Types 5, 2, 8 |
| `Low_Energy` | 2 | Type 6 only |
| `Low_SelfEsteem` | 2 | Type 2 only |
| `Education_Level` | 3 | Type 2 only |

Together these cover all 12 classes — tree models simply memorized them instead of learning any real pattern.

### Proof of Leakage (Orange)
Verified using **Select Rows → Distributions** in Orange:
- Set a single filter (e.g., `Symptoms = 2`)
- Distributions widget shows **one single bar** at exactly one Depression_Type
- Orange confirms: e.g., `38 | 1` = 38 rows, all belonging to 1 unique class

---

## 🔬 Experimental Setup

| Item | Detail |
|------|--------|
| Visual tool | Orange Data Mining (workflow: `classification.ows`) |
| Python library | scikit-learn |
| Models | Random Forest, Logistic Regression, SVM |
| Evaluation | 5-Fold Cross Validation |
| Train/Test split | 80% / 20% (stratified) |
| Feature scaling | StandardScaler applied for LR and SVM |

---

## 📈 Results

### Orange — Before vs After Leakage Removal

**Before (Invalid — all features included):**
| Model | AUC | CA | F1 | MCC |
|-------|-----|----|----|-----|
| Tree | 1.000 | 1.000 | 1.000 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.933 | 0.744 | 0.739 | 0.684 |
| Logistic Regression | 0.914 | 0.723 | 0.715 | 0.654 |

**After (Valid — 6 leaky features removed):**
| Model | AUC | CA | F1 | MCC |
|-------|-----|----|----|-----|
| Tree | 0.993 | 0.985 | 0.985 | 0.981 |
| Random Forest | 0.999 | 0.961 | 0.962 | 0.952 |
| Logistic Regression | 0.836 | 0.577 | 0.556 | 0.463 |
| Naive Bayes | 0.852 | 0.583 | 0.577 | 0.484 |

### Python — Final Tuned Model Comparison

| Model | Test Accuracy | Test F1 | Train F1 | Gap |
|-------|--------------|---------|----------|-----|
| **Random Forest (Tuned)** | **94.94%** | **0.9494** | 0.9793 | 0.030 ✅ |
| Logistic Regression (Tuned) | 80.24% | 0.8024 | 0.8890 | 0.087 ✅ |
| SVM | 79.50% | 0.7883 | — | Stable |

> **Best Model: Random Forest (Tuned)** — highest accuracy, smallest train-test gap, most consistent across all 12 classes.

---

## 🏆 Model Comparison Summary

- **Random Forest** is the top performer — ensemble trees naturally handle non-linear boundaries in behavioral/psychological survey data. Small train-test gap (0.030) confirms it generalizes well without overfitting.
- **Logistic Regression** improved dramatically from 54.9% → 80.2% F1 after polynomial feature engineering and strong regularization (C=0.05) — proving linear models can compete with proper preprocessing.
- **SVM** delivered solid baseline performance with minimal effort — feature scaling alone was sufficient, no tuning needed.

---

## 📝 Key Takeaways

1. **Data integrity before modeling** — 6 leaky features caused 100% scores; removing them was essential for valid results
2. **Tree-based models suit this data** — non-linear decision boundaries make Random Forest the natural best choice
3. **F1 Score over accuracy** — with 12 imbalanced classes, raw accuracy alone is misleading
4. **Linear models need preprocessing** — polynomial features and regularization are non-negotiable for LR on non-linear problems
5. **Synthetic data has limits** — results may not generalize to real clinical populations; acknowledge this in any academic use

---

## 🚀 How to Run

### Orange Workflow
1. Open Orange Data Mining
2. Load `classification.ows`
3. Point the File widget to `data/Mental_Health_Classification.csv`
4. Ensure these 6 columns are set to **Skip:**
   `Symptoms`, `Your overeating level`, `Coping_Methods`, `Low_Energy`, `Low_SelfEsteem`, `Education_Level`

### Python Notebook
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook
```
Open `model_trained_der ver/Train_model_Dear_ver.ipynb` and run all cells.