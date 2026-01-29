"""
Trains & compares several ML models on your landmark dataset.
Compares performance on validation + test set.

Run after you have:
  train.csv
  val.csv
  test.csv
"""
import sys
sys.path.append('D:/University/projectS/hand_tracking')
print("****************** ADDED ROOT DIRECTORY ******************")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import os
# import yaml
import warnings
from src.training.config import cfg

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ─────────────────────────────────────────────────────
model_output = 'models'
os.makedirs(model_output, exist_ok=True)

RANDOM_STATE = cfg['SEED']

# ── Models ─────────────────────────────────────────────────────────────
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
}

# ── Data loading ──────────────────────────────────────────────────────
def load_split(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y


print("Loading datasets...")
X_train, y_train = load_split("dataset/train.csv")
X_val,   y_val   = load_split("dataset/val.csv")
X_test,  y_test  = load_split("dataset/test.csv")

print(f"Train: {X_train.shape[0]:4d} samples")
print(f"Val  : {X_val.shape[0]:4d} samples")
print(f"Test : {X_test.shape[0]:4d} samples")

# ── Preprocessing ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ── Training & Evaluation ─────────────────────────────────────────────
results = []

print("\n" + "="*70)
print("Training models...")
print("="*70)

for name, model in MODELS.items():
    print(f"→ {name:<18} ... ", end="", flush=True)
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    acc_val  = accuracy_score(y_val,  y_pred_val)  * 100
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    
    results.append({
        "Model": name,
        "Val Acc (%)":  round(acc_val,  2),
        "Test Acc (%)": round(acc_test, 2)
    })
    print(f"Val: {acc_val:5.2f}% | Test: {acc_test:5.2f}%")
    
    joblib.dump(model, f"{model_output}/{name}.joblib")
    print(f'Model {name} saved!')

joblib.dump(scaler, f"{model_output}/scaler.joblib")
print("Scaler saved")

# ── Show results table ────────────────────────────────────────────────
print("\n" + "="*70)
print("Final Results Comparison")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test Acc (%)", ascending=False)
print(results_df.to_string(index=False))

# ── Best model on test set ────────────────────────────────────────────
best_model_name = results_df.iloc[0]["Model"]
best_test_acc   = results_df.iloc[0]["Test Acc (%)"]

print(f"\nBest model (on test set): {best_model_name} → {best_test_acc:.2f}%")

# Optional: detailed report for the best model
print(f"\nClassification Report for {best_model_name} (test set):")
best_model = MODELS[best_model_name]
best_model.fit(X_train, y_train)  # re-fit if needed
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, digits=3))