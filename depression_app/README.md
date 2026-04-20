# Depression Type Classifier — Flask Web App

A local Python web app for classifying 12 types of depression using
your own trained machine learning models.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# Optional: XGBoost support
pip install xgboost

# 2. Run the app
python app.py

# 3. Open in browser
# http://localhost:5000
```

## Supported model formats

| Format | Framework | Extension |
|--------|-----------|-----------|
| Pickle | scikit-learn (RF, SVM, LR, etc.) | `.pkl` |
| Joblib | scikit-learn | `.joblib` |
| JSON   | XGBoost | `.json` |

## Input features (15 total)

| Feature | Type | Encoding |
|---------|------|----------|
| Gender | Categorical | 0=Male, 1=Female |
| Age | Numeric | — |
| Employment_Status | Categorical | 0=Unemployed, 1=Student, 2=Employed, 3=Self-employed, 4=Other |
| Search_Depression_Online | Binary | 0=No, 1=Yes |
| Worsening_Depression | Binary | 0=No, 1=Yes |
| How_many_times_you_eat | Ordinal | 0=≤2, 1=3, 2=4–5, 3=>5 meals |
| SocialMedia_Hours | Numeric | hours/day |
| SocialMedia_WhileEating | Ordinal | 0=Never, 1=Rarely, 2=Often, 3=Always |
| Sleep_Hours | Numeric | hours/day |
| Nervous_Level | Ordinal | 0=None, 1=Mild, 2=Moderate, 3=Severe |
| Depression_Score | Numeric | 0–30 scale |
| Self_Harm | Binary | 0=No history, 1=History |
| Mental_Health_Support | Binary | 0=No, 1=Yes |
| Suicide_Attempts | Ordinal | 0=None, 1=Once, 2=Twice, 3=Three+ |

## Output — 12 depression types

| Class | Abbr | Name |
|-------|------|------|
| 0 | None | No clinically significant depression |
| 1 | MLD | Minimal / Mild depression |
| 2 | MOD | Moderate depression |
| 3 | MSD | Moderately-severe depression |
| 4 | SEV | Severe depression |
| 5 | PDD | Persistent depressive disorder (Dysthymia) |
| 6 | SAD | Seasonal affective pattern |
| 7 | PPD | Peripartum / Postpartum depression |
| 8 | BDE | Bipolar-related depressive episode |
| 9 | SRD | Situational / Reactive depression |
| 10 | PSY | Psychotic depression |
| 11 | OSDD | Other specified depressive disorder |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/upload_model` | Upload model file (form: slot, model_file) |
| POST | `/remove_model` | Remove a model slot (JSON: {slot}) |
| GET | `/models_status` | List loaded models |
| POST | `/predict` | Run prediction (JSON: feature dict) |

## Notes

- Models are stored in the `models/` directory while the server is running.
- Up to 3 models can be loaded simultaneously; predictions show per-model
  results plus an ensemble majority-vote consensus.
- If a model supports `predict_proba`, the full class probability
  distribution is shown as a bar chart.
