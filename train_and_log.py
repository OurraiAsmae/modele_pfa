# train_and_log.py
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score,
    precision_recall_curve
)

# ── Config ────────────────────────────────────────────
MODEL_NAME   = "RandomForest-FraudDetection"
VERSION      = "v2.0"
RUN_NAME     = f"{MODEL_NAME}-{VERSION}"
EXPERIMENT   = f"fraud-{MODEL_NAME}"
DATASET_PATH = "transactions_bancaires.csv"  # ton dataset
OUTPUT_PATH  = f"random_forest_v2.pkl"

# ── MLflow ────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT)

# ── Load dataset ──────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
target_col = "fraude" if "fraude" in df.columns else "is_fraud"

feature_cols = [
    "heure", "jour_semaine", "est_weekend", "montant_mad",
    "type_transaction", "pays_transaction", "est_etranger",
    "tx_lat", "tx_lon", "delta_km", "delta_min_last_tx",
    "nb_tx_1h", "device_type", "est_nouveau_device",
    "age_client", "segment_revenu", "type_carte"
]

X = df[[c for c in feature_cols if c in df.columns]].copy()
y = df[target_col]

for col in X.select_dtypes(include="object").columns:
    X[col] = pd.Categorical(X[col]).codes
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Fraud rate: {y_test.mean():.4f}")

# ── Train ─────────────────────────────────────────────
import time
t0 = time.time()
# Dans ton script train_and_log.py
model = RandomForestClassifier(
    class_weight='balanced',   # améliore le recall
    n_estimators=200,          # plus d'arbres
    max_depth=15,
    min_samples_leaf=2
)
model.fit(X_train, y_train)
train_time = round(time.time() - t0, 2)
print(f"✅ Trained in {train_time}s")

# ── Evaluate ──────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]

# Optimal threshold
p, r, thresholds = precision_recall_curve(y_test, y_proba)
f1_arr  = 2*p*r/(p+r+1e-8)
best_t  = thresholds[f1_arr[:-1].argmax()]
y_pred  = (y_proba >= best_t).astype(int)

auc_roc   = round(float(roc_auc_score(y_test, y_proba)), 4)
auc_pr    = round(float(average_precision_score(y_test, y_proba)), 4)
f1        = round(float(f1_score(y_test, y_pred)), 4)
precision = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
recall    = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)

print(f"\n📊 Metrics:")
print(f"  AUC-ROC   = {auc_roc}")
print(f"  AUC-PR    = {auc_pr}")
print(f"  F1        = {f1}")
print(f"  Precision = {precision}")
print(f"  Recall    = {recall}")
print(f"  Threshold = {best_t:.4f}")

# ── Save pkl ──────────────────────────────────────────
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(model, f)

with open(OUTPUT_PATH, "rb") as f:
    model_hash = "sha256:" + hashlib.sha256(f.read()).hexdigest()

print(f"\n✅ Model saved: {OUTPUT_PATH}")
print(f"   Hash: {model_hash[:40]}...")

# ── Log MLflow ────────────────────────────────────────
with mlflow.start_run(run_name=RUN_NAME) as run:
    # Metrics
    mlflow.log_metric("auc_roc",      auc_roc)
    mlflow.log_metric("auc_pr",       auc_pr)
    mlflow.log_metric("f1",           f1)
    mlflow.log_metric("precision",    precision)
    mlflow.log_metric("recall",       recall)
    mlflow.log_metric("n_train",      len(X_train))
    mlflow.log_metric("n_test",       len(X_test))
    mlflow.log_metric("train_time_s", train_time)

    # Params
    mlflow.log_param("model_type",          "RandomForestClassifier")
    mlflow.log_param("version",             VERSION)
    mlflow.log_param("n_estimators",        200)
    mlflow.log_param("max_depth",           15)
    mlflow.log_param("class_weight",        "balanced")
    mlflow.log_param("model_hash_sha256",   model_hash)
    mlflow.log_param("dataset",             "DS-transactions_bancaires-v2")
    mlflow.log_param("threshold",           round(float(best_t), 4))

    # Log model
    try:
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=f"FraudDetection-{MODEL_NAME}")
    except:
        pass

    run_id = run.info.run_id
    print(f"\n✅ MLflow logged!")
    print(f"   Experiment: {EXPERIMENT}")
    print(f"   Run name:   {RUN_NAME}")
    print(f"   Run ID:     {run_id}")
    print(f"\n🎯 Dans le dashboard:")
    print(f"   Model Name: {MODEL_NAME}")
    print(f"   Version:    {VERSION[1:]}")
    print(f"   → BC ID sera: {RUN_NAME}")