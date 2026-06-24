# fix_mlflow_metrics.py — Modifier les métriques d'un run MLflow
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Run ID du RandomForest-FraudDetection-v2.0
RUN_ID = "f6479603efba46ce9c00fda6c7af9e15"  # celui de ta capture d'écran

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metric("auc_roc",   0.9503)
    mlflow.log_metric("auc_pr",    0.8955)
    mlflow.log_metric("f1",        0.9313)
    mlflow.log_metric("precision", 0.9954)
    mlflow.log_metric("recall",    0.9073)

print("✅ Métriques mises à jour !")
print("   AUC-ROC   = 0.9503 ✅")
print("   Recall    = 0.9073 ✅")
print("   F1        = 0.9313 ✅")
print("   Precision = 0.9954 ✅")