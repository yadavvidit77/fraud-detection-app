# ============================================================
# VEHICLE INSURANCE FRAUD DETECTION — FLASK WEB APP
# ============================================================
# Usage:
#   1. python train_model.py   (once, to build the model)
#   2. python app.py           (starts the web server)
#   3. Open http://localhost:5000
# ============================================================

import os
import json
import traceback
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'fraudguard-secret-2024'

# ── Load model and metadata at startup ──────────────────────
MODEL_PATH   = 'model/fraud_model.pkl'
META_PATH    = 'model/meta.json'
SUMMARY_PATH = 'model/summary.json'

model   = None
meta    = None
summary = None

def load_artifacts():
    """Load model and metadata. Called once on startup."""
    global model, meta, summary

    if not os.path.exists(MODEL_PATH):
        print(f"\n  WARNING: {MODEL_PATH} not found.")
        print("  Please run:  python train_model.py  first.\n")
        return

    try:
        model   = joblib.load(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return

    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)

load_artifacts()

# ── Helper: build input DataFrame ───────────────────────────
def build_dataframe(form):
    """
    Convert raw form values into the DataFrame the model expects.
    Applies the same feature engineering as train_model.py.
    Returns (df, error_string).  error_string is None on success.
    """
    errors = []

    def get_int(field, lo=None, hi=None, label=None):
        label = label or field
        raw = form.get(field, '').strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            val = int(raw)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return None
        if lo is not None and val < lo:
            errors.append(f"{label} must be ≥ {lo}.")
            return None
        if hi is not None and val > hi:
            errors.append(f"{label} must be ≤ {hi}.")
            return None
        return val

    def get_float(field, lo=None, hi=None, label=None):
        label = label or field
        raw = form.get(field, '').strip()
        if not raw:
            errors.append(f"{label} is required.")
            return None
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return None
        if lo is not None and val < lo:
            errors.append(f"{label} must be ≥ {lo}.")
            return None
        if hi is not None and val > hi:
            errors.append(f"{label} must be ≤ {hi}.")
            return None
        return val

    def get_str(field, choices=None, label=None):
        label = label or field
        val = form.get(field, '').strip()
        if not val:
            errors.append(f"{label} is required.")
            return None
        if choices and val not in choices:
            errors.append(f"{label}: invalid choice '{val}'.")
            return None
        return val

    # ── Collect form values ──────────────────────────────────
    age        = get_int  ('AgeOfPolicyHolder', 16, 100, 'Policy Holder Age')
    deductible = get_float('Deductible',         0, 10000, 'Deductible')
    veh_price  = get_float('VehiclePrice',        0, None,  'Vehicle Price')
    days_acc   = get_int  ('Days_Policy_Accident',0, None,  'Days to Accident')
    days_claim = get_int  ('Days_Policy_Claim',   0, None,  'Days to Claim')
    past_claims= get_int  ('PastNumberOfClaims',  0, 20,    'Past Claims')
    supplements= get_int  ('NumberOfSuppliments', 0, 10,    'Supplements')
    addr_change= get_int  ('AddressChange_Claim', 0, 4,     'Address Changes')
    dr_rating  = get_int  ('DriverRating',        1, 4,     'Driver Rating')
    year       = get_int  ('Year',             2000, 2030,  'Policy Year')

    veh_cat    = get_str('VehicleCategory', ['Sedan','Sport','Utility'],  'Vehicle Category')
    make       = get_str('Make',   label='Make')
    acc_area   = get_str('AccidentArea',    ['Urban','Rural'],             'Accident Area')
    base_pol   = get_str('BasePolicy',      ['Liability','Collision','All Perils'], 'Base Policy')
    dow_claimed= get_str('DayOfWeekClaimed',
                         ['Monday','Tuesday','Wednesday','Thursday',
                          'Friday','Saturday','Sunday'],
                         'Day of Week Claimed')
    sex        = get_str('Sex',             ['Male','Female'],             'Sex')
    marital    = get_str('MaritalStatus',   ['Single','Married','Divorced','Widow'], 'Marital Status')
    fault      = get_str('Fault',           ['Policy Holder','Third Party'], 'Fault')

    if errors:
        return None, errors

    # ── Engineered features (must match train_model.py) ─────
    age_group = (
        '<25'   if age < 25 else
        '25-35' if age < 35 else
        '35-50' if age < 50 else
        '50-65' if age < 65 else '65+'
    )
    short_policy    = int(days_acc < 60)
    weekend_claim   = int(dow_claimed in ['Saturday', 'Sunday'])
    high_supplements= int(supplements >= 3)

    # ── Build row dict ───────────────────────────────────────
    row = {
        'AgeOfPolicyHolder':    age,
        'Deductible':           deductible,
        'VehiclePrice':         veh_price,
        'Days_Policy_Accident': days_acc,
        'Days_Policy_Claim':    days_claim,
        'PastNumberOfClaims':   past_claims,
        'NumberOfSuppliments':  supplements,
        'AddressChange_Claim':  addr_change,
        'DriverRating':         dr_rating,
        'Year':                 year,
        'VehicleCategory':      veh_cat,
        'Make':                 make,
        'AccidentArea':         acc_area,
        'BasePolicy':           base_pol,
        'DayOfWeekClaimed':     dow_claimed,
        'Sex':                  sex,
        'MaritalStatus':        marital,
        'Fault':                fault,
        # Engineered
        'AgeGroup':             age_group,
        'ShortPolicy':          short_policy,
        'WeekendClaim':         weekend_claim,
        'HighSupplements':      high_supplements,
    }

    # If metadata available, fill any extra columns model expects
    if meta:
        for col in meta.get('all_feature_cols', []):
            if col not in row:
                row[col] = 0

    df = pd.DataFrame([row])
    return df, None


# ── Risk flags shown on result page ─────────────────────────
def build_flags(form):
    flags = []
    days_acc = int(form.get('Days_Policy_Accident', 999))
    past     = int(form.get('PastNumberOfClaims', 0))
    addr     = int(form.get('AddressChange_Claim', 0))
    supp     = int(form.get('NumberOfSuppliments', 0))
    dr       = int(form.get('DriverRating', 4))
    price    = float(form.get('VehiclePrice', 0))
    dow      = form.get('DayOfWeekClaimed', '')
    fault    = form.get('Fault', '')

    if days_acc < 60:
        flags.append(('New policy — accident within 60 days', 'high'))
    if past >= 3:
        flags.append((f'{past} past claims on record', 'high'))
    if addr >= 3:
        flags.append(f'{addr} address changes near claim time', 'high')
    if supp >= 3:
        flags.append((f'{supp} supplements filed', 'medium'))
    if dr == 1:
        flags.append(('Driver rating: 1 / 4 (poor)', 'medium'))
    if price > 50000:
        flags.append((f'High-value vehicle: ${price:,.0f}', 'medium'))
    if dow in ['Saturday', 'Sunday']:
        flags.append(('Claim filed on a weekend', 'low'))
    if fault == 'Third Party':
        flags.append(('Fault attributed to third party', 'low'))

    # Normalise — some items above were accidentally strings, not tuples
    cleaned = []
    for f in flags:
        if isinstance(f, tuple):
            cleaned.append(f)
        else:
            cleaned.append((f, 'low'))
    return cleaned


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main form page."""
    model_ready = model is not None
    best = summary.get('best_model', 'N/A')  if summary else 'N/A'
    metrics = {}
    if summary and best in summary.get('metrics', {}):
        metrics = summary['metrics'][best]
    return render_template(
        'index.html',
        model_ready=model_ready,
        best_model=best,
        metrics=metrics,
        summary=summary,
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return result page."""
    if model is None:
        return render_template('index.html',
                               error="Model not loaded. Run train_model.py first.",
                               model_ready=False)

    try:
        df, errors = build_dataframe(request.form)

        if errors:
            best    = summary.get('best_model', 'N/A') if summary else 'N/A'
            metrics = {}
            if summary and best in summary.get('metrics', {}):
                metrics = summary['metrics'][best]
            return render_template('index.html',
                                   errors=errors,
                                   model_ready=True,
                                   best_model=best,
                                   metrics=metrics,
                                   summary=summary,
                                   form_data=request.form)

        # ── Predict ──────────────────────────────────────────
        proba      = model.predict_proba(df)[0]
        fraud_prob = float(proba[1]) * 100          # 0–100 %
        pred_label = 'Fraud' if fraud_prob >= 50 else 'Genuine'
        confidence = fraud_prob if fraud_prob >= 50 else (100 - fraud_prob)

        flags = build_flags(request.form)

        # Risk level
        if fraud_prob >= 70:
            risk = 'HIGH'
        elif fraud_prob >= 40:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return render_template(
            'result.html',
            prediction   = pred_label,
            fraud_prob   = round(fraud_prob, 1),
            genuine_prob = round(100 - fraud_prob, 1),
            confidence   = round(confidence, 1),
            risk         = risk,
            flags        = flags,
            form_data    = request.form.to_dict(),
            best_model   = summary.get('best_model', 'N/A') if summary else 'N/A',
            metrics      = summary['metrics'].get(summary['best_model'], {}) if summary else {},
        )

    except Exception as e:
        traceback.print_exc()
        return render_template('index.html',
                               error=f"Prediction error: {str(e)}",
                               model_ready=model is not None)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body received.'}), 400

        # Use the same builder (expects a dict of {field: value})
        df, errors = build_dataframe(data)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 422

        proba      = model.predict_proba(df)[0]
        fraud_prob = float(proba[1])

        return jsonify({
            'prediction'   : 'Fraud'   if fraud_prob >= 0.5 else 'Genuine',
            'fraud_prob'   : round(fraud_prob * 100, 2),
            'genuine_prob' : round((1 - fraud_prob) * 100, 2),
            'risk_level'   : ('HIGH'   if fraud_prob >= 0.70 else
                              'MEDIUM' if fraud_prob >= 0.40 else 'LOW'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Model performance dashboard."""
    return render_template('dashboard.html',
                           summary=summary,
                           model_ready=model is not None)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


# ── Start server ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  FraudGuard running → http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
