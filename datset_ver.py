# Ajoute ce code AVANT le training dans train_xgboost.py
# pour analyser la qualité du dataset

import pandas as pd
import numpy as np

df = pd.read_csv("transactions_bancaires.csv")
target_col = "fraude" if "fraude" in df.columns else "is_fraud"

print("=== ANALYSE DATASET ===")
print(f"Total lignes    : {len(df):,}")
print(f"Fraudes         : {df[target_col].sum():,} ({df[target_col].mean()*100:.2f}%)")
print(f"Légitimes       : {(df[target_col]==0).sum():,}")
print()

# Corrélation features avec target
feature_cols = [
    "heure","jour_semaine","est_weekend","montant_mad",
    "type_transaction","pays_transaction","est_etranger",
    "tx_lat","tx_lon","delta_km","delta_min_last_tx",
    "nb_tx_1h","device_type","est_nouveau_device",
    "age_client","segment_revenu","type_carte"
]
X = df[[c for c in feature_cols if c in df.columns]].copy()
for col in X.select_dtypes(include="object").columns:
    X[col] = pd.Categorical(X[col]).codes
X = X.fillna(0)

corr = X.corrwith(df[target_col]).abs().sort_values(ascending=False)
print("=== CORRÉLATION FEATURES / FRAUDE ===")
for feat, val in corr.items():
    bar = "█" * int(val * 40)
    print(f"  {feat:25s} {val:.4f}  {bar}")

print()
# Moyenne par classe
print("=== MOYENNE PAR CLASSE (fraude vs légitime) ===")
for col in corr.index[:5]:
    fraud_mean = df[df[target_col]==1][col].mean()
    legit_mean = df[df[target_col]==0][col].mean()
    print(f"  {col:25s} fraude={fraud_mean:.3f}  légitime={legit_mean:.3f}")