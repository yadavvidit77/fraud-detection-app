# ================================================
# VEHICLE INSURANCE FRAUD DETECTION SYSTEM
# Complete ML Project in Single File
# Fully Runnable in Google Colab
# ================================================

# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚗 VEHICLE INSURANCE FRAUD DETECTION SYSTEM")
print("=" * 60)

# ================================================
# 1. DATA PREPROCESSING
# ================================================
print("\n📊 1. DATA PREPROCESSING")

# Load dataset (Download fraud_oracle.csv from Kaggle or use sample data generation)
df = pd.read_csv('/Users/vidit/Downloads/fraud_oracle.csv')
print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Handle missing values
print("\n🔧 Handling missing values...")
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f"✅ Removed {initial_rows - len(df)} duplicates")

# Drop unnecessary columns
df.drop(['PolicyNumber', 'RepNumber'], axis=1, inplace=True, errors='ignore')
print(f"✅ Dropped unnecessary columns. Final shape: {df.shape}")

# ================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================
print("\n📈 2. EXPLORATORY DATA ANALYSIS")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('🚗 Insurance Fraud Analysis Dashboard', fontsize=16, fontweight='bold')

# --------------------------------
# 1. Fraud distribution
# --------------------------------
fraud_count = df['FraudFound_P'].value_counts()

axes[0,0].pie(
    fraud_count.values,
    labels=['Not Fraud', 'Fraud'],
    autopct='%1.1f%%',
    colors=['#2ecc71', '#e74c3c'],
    explode=[0.1, 0]
)
axes[0,0].set_title('Fraud vs Non-Fraud Distribution')

# --------------------------------
# 2. Fraud by VehicleCategory
# --------------------------------
sns.countplot(data=df, x='VehicleCategory', hue='FraudFound_P', ax=axes[0,1])
axes[0,1].set_title('Fraud by Vehicle Category')
axes[0,1].tick_params(axis='x', rotation=45)

# --------------------------------
# 3. Age Conversion (FIXED)
# --------------------------------
def convert_age(age):
    try:
        if isinstance(age, str):
            if 'to' in age:
                parts = age.split(' to ')
                return (int(parts[0]) + int(parts[1])) / 2
            elif '-' in age:
                parts = age.split('-')
                return (int(parts[0]) + int(parts[1])) / 2
            elif 'over' in age:
                return 70  # approx
        return float(age)
    except:
        return np.nan

df['AgeOfPolicyHolder'] = df['AgeOfPolicyHolder'].apply(convert_age)

# Fill missing values
df['AgeOfPolicyHolder'].fillna(df['AgeOfPolicyHolder'].median(), inplace=True)

# Create Age Group
df['Age_Group'] = pd.cut(
    df['AgeOfPolicyHolder'],
    bins=[18, 30, 42, 54, 66, 100],
    labels=['18-30', '31-42', '43-54', '55-66', '67+']
)

# --------------------------------
# 4. Fraud by Age Group (NEW FIX)
# --------------------------------
sns.countplot(data=df, x='Age_Group', hue='FraudFound_P', ax=axes[0,2])
axes[0,2].set_title('Fraud by Age Group')

# --------------------------------
# 5. Fraud by Accident Area
# --------------------------------
sns.countplot(data=df, x='AccidentArea', hue='FraudFound_P', ax=axes[1,0])
axes[1,0].set_title('Fraud by Accident Area')

# --------------------------------
# 6. Correlation Heatmap (FIXED)
# --------------------------------
# Ensure target is numeric
df['FraudFound_P'] = df['FraudFound_P'].astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns

corr_matrix = df[numeric_cols].corr()

sns.heatmap(
    corr_matrix[['FraudFound_P']]
    .sort_values(by='FraudFound_P', key=abs, ascending=False),
    annot=True,
    cmap='RdBu_r',
    center=0,
    ax=axes[1,1]
)

axes[1,1].set_title('Correlation with Fraud')

# --------------------------------
# 7. Hide unused subplot
# --------------------------------
axes[1,2].axis('off')

# --------------------------------
# FINAL
# --------------------------------
plt.tight_layout()
plt.show()

# ================================================
# 3. SPLIT DATA
# ================================================
print("\n✂️ Splitting data...")

X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Columns
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("📋 Categorical:", len(cat_cols))
print("📋 Numeric:", len(num_cols))

# ================================================
# 4. PIPELINE
# ================================================
print("\n🔄 Building pipeline...")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ))
])

# ================================================
# 5. TRAIN MODEL
# ================================================
print("\n🎯 Training model...")

model_pipeline.fit(X_train, y_train)

print("✅ Training complete!")

# ================================================
# 6. EVALUATION
# ================================================
print("\n📊 Evaluating model...")

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.3f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")

# ROC Curve
plt.subplot(1,2,2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
plt.title("ROC Curve")

plt.tight_layout()
plt.show()

print(f"🎯 ROC-AUC: {roc_auc:.3f}")

# ================================================
# 7. SAVE MODEL
# ================================================
joblib.dump(model_pipeline, "fraud_model.pkl")
print("💾 Model saved as fraud_model.pkl")

# ================================================
# 8. REAL-TIME DETECTOR CLASS
# ================================================
class FraudDetector:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, data):
        df = pd.DataFrame([data])
        prob = self.pipeline.predict_proba(df)[0][1]
        pred = self.pipeline.predict(df)[0]

        if prob < 0.5:
            risk = "LOW 🟢"
        elif prob <= 0.8:
            risk = "MEDIUM 🟡"
        else:
            risk = "HIGH 🔴"

        return {
            "prediction": "FRAUD" if pred == 1 else "NOT FRAUD",
            "probability": round(prob * 100, 2),
            "risk": risk
        }

# ================================================
# 9. TEST SAMPLE
# ================================================
detector = FraudDetector(model_pipeline)

sample = X_test.iloc[0].to_dict()
result = detector.predict(sample)

print("\n🚨 Sample Prediction:")
print(result)

# ================================================
# 🎉 DONE
# ================================================
print("\n🎉 PROJECT COMPLETE!")