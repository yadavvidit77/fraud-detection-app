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
# ── Load model and metadata at startup ──────────────────────

MODEL_PATH = 'model/fraud_model.pkl'
META_PATH = 'model/meta.json'
SUMMARY_PATH = 'model/summary.json'

model = None
required_cols = []
meta = None
summary = None

def load_artifacts():
    global model, required_cols, meta, summary

    # -----------------------------
    # Load model
    # -----------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("👉 Run: python train_model.py")
        return

    try:
        saved = joblib.load(MODEL_PATH)

        # Handle both formats safely
        if isinstance(saved, dict):
            model = saved.get("model")
            required_cols = saved.get("columns", [])
        else:
            model = saved
            required_cols = []

        print("✅ Model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # -----------------------------
    # Load metadata (optional)
    # -----------------------------
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)

# Call loader
load_artifacts()
 
# ── Helper: build input DataFrame ───────────────────────────
def build_dataframe(form):
    errors = []

    # ---------------------------
    # SAFE INPUT FUNCTIONS
    # ---------------------------
    def get_int(field):
        try:
            return int(form.get(field, 0))
        except:
            errors.append(f"{field} must be integer")
            return 0

    def get_float(field):
        try:
            return float(form.get(field, 0))
        except:
            errors.append(f"{field} must be number")
            return 0.0

    def get_str(field):
        val = form.get(field)
        return str(val) if val is not None else "Unknown"

    # ---------------------------
    # NUMERIC INPUTS
    # ---------------------------
    age         = get_int('AgeOfPolicyHolder')
    deductible  = get_float('Deductible')
    veh_price   = get_float('VehiclePrice')
    days_acc    = get_int('Days_Policy_Accident')
    days_claim  = get_int('Days_Policy_Claim')
    past_claims = get_int('PastNumberOfClaims')
    supplements = get_int('NumberOfSuppliments')
    addr_change = get_int('AddressChange_Claim')
    dr_rating   = get_int('DriverRating')
    year        = get_int('Year')

    # ---------------------------
    # CATEGORICAL INPUTS
    # ---------------------------
    veh_cat  = get_str('VehicleCategory')
    make     = get_str('Make')
    acc_area = get_str('AccidentArea')
    base_pol = get_str('BasePolicy')
    dow      = get_str('DayOfWeekClaimed')
    sex      = get_str('Sex')
    marital  = get_str('MaritalStatus')
    fault    = get_str('Fault')

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    age_group = (
        '<25' if age < 25 else
        '25-35' if age < 35 else
        '35-50' if age < 50 else
        '50-65' if age < 65 else '65+'
    )

    row = {
        'AgeOfPolicyHolder': age,
        'Deductible': deductible,
        'VehiclePrice': veh_price,
        'Days_Policy_Accident': days_acc,
        'Days_Policy_Claim': days_claim,
        'PastNumberOfClaims': past_claims,
        'NumberOfSuppliments': supplements,
        'AddressChange_Claim': addr_change,
        'DriverRating': dr_rating,
        'Year': year,

        'VehicleCategory': veh_cat,
        'Make': make,
        'AccidentArea': acc_area,
        'BasePolicy': base_pol,
        'DayOfWeekClaimed': dow,
        'Sex': sex,
        'MaritalStatus': marital,
        'Fault': fault,

        'AgeGroup': age_group,
        'ShortPolicy': int(days_acc < 60),
        'WeekendClaim': int(dow in ['Saturday', 'Sunday']),
        'HighSupplements': int(supplements >= 3),
    }

    df = pd.DataFrame([row])

    # =========================================================
    # 🔥 STEP 1: ADD ALL REQUIRED COLUMNS FIRST
    # =========================================================
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0   # safest default

    # =========================================================
    # 🔥 STEP 2: FIX NUMERIC TYPES
    # =========================================================
    numeric_cols = [
        'AgeOfPolicyHolder', 'Deductible', 'VehiclePrice',
        'Days_Policy_Accident', 'Days_Policy_Claim',
        'PastNumberOfClaims', 'NumberOfSuppliments',
        'AddressChange_Claim', 'DriverRating', 'Year',
        'ShortPolicy', 'WeekendClaim', 'HighSupplements'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # =========================================================
    # 🔥 STEP 3: HANDLE NaN
    # =========================================================
    df.fillna(0, inplace=True)

    # =========================================================
    # 🔥 STEP 4: FIX CATEGORICAL TYPES
    # =========================================================
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].astype(str)

    # =========================================================
    # 🔥 STEP 5: MATCH TRAINING COLUMN ORDER
    # =========================================================
    df = df[required_cols]

    return df, errors
 
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
        flags.append((f'{addr} address changes near claim time', 'high'))  # fixed: was missing outer ()
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
 
    return flags
 
 
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
    app.run(debug=True, host='0.0.0.0', port=5000)
 
