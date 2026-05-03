# Classification of Twelve Types of Depression

A machine learning project that classifies 12 clinically recognized depression subtypes using demographic, behavioral, and psychological survey data.

---

## File Structure

```
├── data/
│   └── Mental_Health_Classification.csv    # Raw dataset (1,998 instances, 21 features)
│
├── models/                                 # Trained model files (.pkl) + training notebook
│   ├── models.ipynb                        # Full training pipeline (RF, LR, SVM)
│   ├── random_forest.pkl                   # Saved Random Forest model
│   ├── logistic_regression.pkl             # Saved Logistic Regression model
│   └── SVM.pkl                             # Saved SVM model
│
├── depression_app/                         # Web application source (deployed on Vercel)
│
├── Model with leak identification.ipynb    # Cross-tabulation analysis and feature removal
├── classification.ows                      # Orange Data Mining workflow
└── README.md
```

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps

1. Clone the repository:

```bash
git clone https://github.com/ppanturat/Classification-of-Twelve-Types-of-Depression.git
cd Classification-of-Twelve-Types-of-Depression
```

2. Place `Mental_Health_Classification.csv` inside the `data/` folder.

3. Run the notebooks in this order:
   - `Model with leak identification.ipynb` — run this first to understand which features were removed and why
   - `models/models.ipynb` — main training and evaluation of all three models

### Web Application

A deployed version is available at: https://classification-of-twelve-types-of-d.vercel.app
