# ============================================================
# VEHICLE INSURANCE FRAUD DETECTION — MODEL TRAINING
# ============================================================
# Run this file ONCE before starting the web app.
# It trains 3 models, picks the best one, and saves it.
# Usage: python train_model.py
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')                      # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# ── Folders ─────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

print("=" * 65)
print("  FRAUDGUARD — Model Training Pipeline")
print("=" * 65)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("\n[1/7] Loading dataset...")

# if not os.path.exists('fraud_oracle.csv'):
#     print("\n  ERROR: fraud_oracle.csv not found!")
#     print("  Download it from:")
#     print("  https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection")
#     print("  and place it in this folder.\n")
#     exit(1)

df = pd.read_csv("/Users/vidit/Downloads/fraud_oracle.csv")
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"  Fraud rate: {df['FraudFound_P'].mean()*100:.1f}%")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n[2/7] Preprocessing...")

# ── Drop columns that leak info or are identifiers ──────────
drop_cols = ['PolicyNumber', 'RepNumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ── Rename target if needed (dataset uses FraudFound_P) ────
if 'FraudFound_P' in df.columns:
    df.rename(columns={'FraudFound_P': 'Fraud'}, inplace=True)

# ── Drop duplicates ──────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Dropped {before - len(df)} duplicates")

# ── Fill missing values ──────────────────────────────────────
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ── Engineer new features ────────────────────────────────────
# Age group (bins)
# ── Engineer new features ────────────────────────────────────

# Age group (bins)
# ============================================================
# 🔥 FEATURE ENGINEERING (CORRECTED)
# ============================================================

# Age conversion + grouping
df['AgeOfPolicyHolder'] = pd.to_numeric(df['AgeOfPolicyHolder'], errors='coerce')
df['AgeOfPolicyHolder'].fillna(df['AgeOfPolicyHolder'].median(), inplace=True)

df['AgeGroup'] = pd.cut(
    df['AgeOfPolicyHolder'],
    bins=[0, 25, 35, 50, 65, 120],
    labels=['<25', '25-35', '35-50', '50-65', '65+']
).astype(str)


# ✅ Convert Days_Policy_Accident (string → numeric)
mapping_days = {
    'none': 0,
    '0 to 15': 10,
    '15 to 30': 22,
    '30 to 60': 45,
    'more than 60': 75
}

df['Days_Policy_Accident'] = df['Days_Policy_Accident'].map(mapping_days)
df['Days_Policy_Accident'].fillna(0, inplace=True)


# (Optional but recommended)
if 'Days_Policy_Claim' in df.columns:
    df['Days_Policy_Claim'] = df['Days_Policy_Claim'].map(mapping_days)
    df['Days_Policy_Claim'].fillna(0, inplace=True)


# ✅ Convert NumberOfSuppliments (string → numeric)
mapping_suppliments = {
    'none': 0,
    '1': 1,
    '2 to 4': 3,
    'more than 4': 5
}

df['NumberOfSuppliments'] = df['NumberOfSuppliments'].map(mapping_suppliments)
df['NumberOfSuppliments'].fillna(0, inplace=True)


# ✅ Final engineered features (now safe)
df['ShortPolicy'] = (df['Days_Policy_Accident'] < 60).astype(int)
df['WeekendClaim'] = df['DayOfWeekClaimed'].isin(['Saturday', 'Sunday']).astype(int)
df['HighSupplements'] = (df['NumberOfSuppliments'] >= 3).astype(int)
# ============================================================
# ✅ FIX NaN ISSUE (ADD HERE)
# ============================================================

# Fill any remaining NaN values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna('Unknown', inplace=True)
    else:
        df[col].fillna(0, inplace=True)

print("Remaining NaN:", df.isnull().sum().sum())

# ============================================================
# 3. FEATURE / TARGET SPLIT
# ============================================================
print("\n[3/7] Splitting features and target...")

TARGET = 'Fraud'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify column types
CAT_COLS = X.select_dtypes(include='object').columns.tolist()
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Numeric features  : {len(NUM_COLS)}")
print(f"  Categorical features: {len(CAT_COLS)}")

# Save column lists for the Flask app
meta = {
    'cat_cols': CAT_COLS,
    'num_cols': NUM_COLS,
    'all_feature_cols': list(X.columns),
    'engineered': ['AgeGroup', 'ShortPolicy', 'WeekendClaim', 'HighSupplements'],
}
with open('model/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  Saved model/meta.json")

# ── Train / test split (stratified) ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================
print("\n[4/7] Building preprocessing pipeline...")

# Numeric: scale
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Categorical: one-hot encode
cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, NUM_COLS),
    ('cat', cat_pipeline, CAT_COLS),
], remainder='drop')

# ============================================================
# 5. TRAIN MULTIPLE MODELS
# ============================================================
print("\n[5/7] Training models (this may take a minute)...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=5,
        learning_rate=0.1, random_state=42
    ),
}

results = {}

for name, clf in models.items():
    print(f"\n  Training: {name}")

    # Full pipeline: preprocess → SMOTE → classifier
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', clf),
    ])

    pipe.fit(X_train, y_train)

    y_pred       = pipe.predict(X_test)
    y_proba      = pipe.predict_proba(X_test)[:, 1]
    acc          = accuracy_score(y_test, y_pred)
    roc          = roc_auc_score(y_test, y_proba)
    f1           = f1_score(y_test, y_pred)
    cv_scores    = cross_val_score(pipe, X_train, y_train, cv=3,
                                   scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'pipeline': pipe,
        'accuracy': acc,
        'roc_auc': roc,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }

    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    ROC-AUC  : {roc:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 6. PICK BEST MODEL AND SAVE
# ============================================================
print("\n[6/7] Selecting best model...")

best_name = max(results, key=lambda k: results[k]['roc_auc'])
best      = results[best_name]

print(f"\n  WINNER: {best_name}")
print(f"    ROC-AUC  : {best['roc_auc']:.4f}")
print(f"    Accuracy : {best['accuracy']*100:.2f}%")
print(f"    F1 Score : {best['f1']:.4f}")

# Full classification report
print(f"\n  Classification Report ({best_name}):")
print(classification_report(
    y_test, best['y_pred'],
    target_names=['Genuine', 'Fraud']
))

# Save the best pipeline
import joblib

joblib.dump({
    "model": best['pipeline'],          # your trained pipeline
    "columns": X.columns.tolist()       # all training columns
}, 'model/fraud_model.pkl')

print("  Saved model/fraud_model.pkl with columns")
# Save a comparison summary so the Flask app can display it
summary = {
    'best_model': best_name,
    'metrics': {
        name: {
            'accuracy': round(r['accuracy'] * 100, 2),
            'roc_auc': round(r['roc_auc'], 4),
            'f1': round(r['f1'], 4),
            'cv_mean': round(r['cv_mean'], 4),
        }
        for name, r in results.items()
    }
}
with open('model/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved model/summary.json")

# ============================================================
# 7. GENERATE CHARTS FOR DASHBOARD
# ============================================================
print("\n[7/7] Generating charts...")

# ── Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Genuine', 'Fraud'],
    yticklabels=['Genuine', 'Fraud'],
    ax=ax
)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=12, pad=10)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('static/charts/confusion_matrix.png', dpi=120)
plt.close()

# ── Model comparison bar chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
names   = list(results.keys())
accs    = [results[n]['accuracy'] * 100 for n in names]
rocs    = [results[n]['roc_auc'] * 100   for n in names]
x       = np.arange(len(names))
w       = 0.35
bars1   = ax.bar(x - w/2, accs, w, label='Accuracy %',  color='#4A90D9')
bars2   = ax.bar(x + w/2, rocs, w, label='ROC-AUC × 100', color='#E67E22')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(50, 105)
ax.set_title('Model Comparison', fontsize=12)
ax.legend()
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                          f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                          f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('static/charts/model_comparison.png', dpi=120)
plt.close()

# ── Feature importance (Random Forest only) ──────────────────
if 'Random Forest' in results:
    rf_pipe  = results['Random Forest']['pipeline']
    rf_model = rf_pipe.named_steps['clf']
    try:
        ohe_names  = (rf_pipe.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['ohe']
                      .get_feature_names_out(CAT_COLS))
        feat_names = NUM_COLS + list(ohe_names)
        importances = rf_model.feature_importances_
        feat_df = (pd.DataFrame({'Feature': feat_names, 'Importance': importances})
                   .sort_values('Importance', ascending=True)
                   .tail(15))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color='#4A90D9')
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=11)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig('static/charts/feature_importance.png', dpi=120)
        plt.close()
        print("  Saved feature importance chart")
    except Exception as e:
        print(f"  (Feature importance chart skipped: {e})")

print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)
print(f"  Best model : {best_name}")
print(f"  Accuracy   : {best['accuracy']*100:.2f}%")
print(f"  ROC-AUC    : {best['roc_auc']:.4f}")
print("\n  Now run:  python app.py")
print("  Then open: http://localhost:5001")
print("=" * 65 + "\n")

