import os
import pickle
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'json'}

loaded_models = {}

FEATURE_ORDER = [
    'Gender', 'Age', 'Employment_Status', 'Search_Depression_Online',
    'Worsening_Depression', 'How many times you eat',
    'SocialMedia_Hours', 'SocialMedia_WhileEating', 'Sleep_Hours',
    'Nervous_Level', 'Depression_Score', 'Self_Harm',
    'Mental_Health_Support', 'Suicide_Attempts'
]

DEPRESSION_TYPES = {
    0:  {"name": "No clinically significant depression",  "abbr": "None"},
    1:  {"name": "Minimal / Mild depression",             "abbr": "MLD"},
    2:  {"name": "Moderate depression",                   "abbr": "MOD"},
    3:  {"name": "Moderately-severe depression",          "abbr": "MSD"},
    4:  {"name": "Severe depression",                     "abbr": "SEV"},
    5:  {"name": "Persistent depressive disorder",        "abbr": "PDD"},
    6:  {"name": "Seasonal affective pattern",            "abbr": "SAD"},
    7:  {"name": "Peripartum / Postpartum depression",    "abbr": "PPD"},
    8:  {"name": "Bipolar-related depressive episode",    "abbr": "BDE"},
    9:  {"name": "Situational / Reactive depression",     "abbr": "SRD"},
    10: {"name": "Psychotic depression",                  "abbr": "PSY"},
    11: {"name": "Other specified depressive disorder",   "abbr": "OSDD"},
}

# Use an absolute path based on the current working directory
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models') 

def load_models_from_folder():
    print("Loading models from folder...")
    # Debugging: Vercel logs will show you if the directory actually exists
    if not os.path.exists(MODELS_DIR):
        print(f"Directory not found at: {MODELS_DIR}")
        return
    print(MODELS_DIR)
    # Filter files
    pkl_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')])
    
    for i, filename in enumerate(pkl_files[:3]):
        path = os.path.join(MODELS_DIR, filename)
        print(path)
        try:
            # Ensure load_model_from_path doesn't require write access
            model, model_type = load_model_from_path(path, filename)
            loaded_models[i] = {
                'model': model, 
                'name': filename, 
                'type': model_type, 
                'path': path
            }
        except Exception as e:
            # This will show up in Vercel's 'Logs' tab
            print(f"Failed to load {filename}: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_from_path(path, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext == 'json':
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(path)
            return model, 'xgboost'
        except ImportError:
            raise RuntimeError("xgboost is not installed on this server.")
    else:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model, 'sklearn'

# Load models at import time so the app works whether run directly or imported by a WSGI server.
load_models_from_folder()

@app.route('/debug')
def debug():
    tmp_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    return jsonify({
        'tmp_files': tmp_files,
        'loaded_models': [info['name'] for info in loaded_models.values()]
    })


@app.route('/')
def index():
    return render_template('index.html',
                           feature_order=FEATURE_ORDER,
                           depression_types=DEPRESSION_TYPES)


@app.route('/models_status', methods=['GET'])
def models_status():
    status = {}
    for slot, info in loaded_models.items():
        status[str(slot)] = {'name': info['name'], 'type': info['type']}
    return jsonify(status)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No input data'}), 400

    # Build feature vector in correct order
    try:
        feature_vec = []
        missing = []
        for feat in FEATURE_ORDER:
            val = data.get(feat)
            if val is None:
                missing.append(feat)
                feature_vec.append(0.0)
            else:
                feature_vec.append(float(val))
        X = np.array([feature_vec])
    except Exception as e:
        return jsonify({'error': f'Feature parsing error: {str(e)}'}), 400

    if not loaded_models:
        return jsonify({'error': 'No models loaded. Please upload at least one model.'}), 400

    results = []
    for slot in sorted(loaded_models.keys()):
        info = loaded_models[slot]
        model = info['model']
        try:
            pred_class = int(model.predict(X)[0])
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = round(float(np.max(proba)) * 100, 1)
                # Build full probability distribution
                proba_dist = {
                    str(i): round(float(p) * 100, 1)
                    for i, p in enumerate(proba)
                }
            else:
                proba_dist = {}

            label_info = DEPRESSION_TYPES.get(pred_class, {"name": "Unknown", "abbr": "?"})
            results.append({
                'slot': slot,
                'model_name': info['name'],
                'model_type': info['type'],
                'predicted_class': pred_class,
                'label': label_info['name'],
                'abbr': label_info['abbr'],
                'confidence': confidence,
                'proba_dist': proba_dist,
                'error': None
            })
        except Exception as e:
            results.append({
                'slot': slot,
                'model_name': info['name'],
                'model_type': info['type'],
                'predicted_class': None,
                'label': None,
                'abbr': None,
                'confidence': None,
                'proba_dist': {},
                'error': str(e)
            })

    # Ensemble: majority vote among successful predictions
    valid = [r for r in results if r['predicted_class'] is not None]
    ensemble = None
    if valid:
        from collections import Counter
        vote_counts = Counter(r['predicted_class'] for r in valid)
        top_class = vote_counts.most_common(1)[0][0]
        label_info = DEPRESSION_TYPES.get(top_class, {"name": "Unknown", "abbr": "?"})
        ensemble = {
            'predicted_class': top_class,
            'label': label_info['name'],
            'abbr': label_info['abbr'],
            'vote_count': vote_counts[top_class],
            'total_models': len(valid)
        }

    return jsonify({
        'results': results,
        'ensemble': ensemble,
        'missing_features': missing,
        'feature_vector': feature_vec
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
