"""
=============================================================
DEEP LEARNING — TENSORFLOW / KERAS
Gouvernance et Traçabilité des Modèles IA — Fraude Bancaire
=============================================================
3 modèles avec TensorFlow 2.x / Keras :

DL-1 : MLP  → réseau dense classique
DL-2 : Autoencoder → détection d'anomalies (légitimes only)
DL-3 : MLP + Attention → interprétabilité native par feature

Installation :
    pip install tensorflow          # CPU
    pip install tensorflow-gpu      # GPU (NVIDIA)

Compatible : TF 2.10+, Python 3.9+
=============================================================
"""

import numpy as np
import pandas as pd
import json, pickle, hashlib, os, warnings, time
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # silence TF logs

import tensorflow as tf
from tensorflow import keras # type: ignore
from keras import layers, Model, Input
from keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("models",     exist_ok=True)
os.makedirs("reports/dl", exist_ok=True)

print("=" * 65)
print(f"  TENSORFLOW / KERAS — v{tf.__version__}")
print(f"  GPU disponible : {len(tf.config.list_physical_devices('GPU'))} device(s)")
print("=" * 65)

# ─────────────────────────────────────────────────────────
# 1. DONNÉES
# ─────────────────────────────────────────────────────────
print("\n[1/8] Chargement et préparation...")

DATA = ("data/transactions_engineered.csv"
        if os.path.exists("data/transactions_engineered.csv")
        else "data/transactions_bancaires.csv")
print(f"   Source : {DATA}")

df = pd.read_csv(DATA)
BASE = ["heure","jour_semaine","est_weekend","montant_mad","est_etranger",
        "delta_km","delta_min_last_tx","nb_tx_1h","est_nouveau_device",
        "age_client","age_compte_jours","ratio_montant_moy","risque_horaire"]
ENG = [f for f in ["log_montant_mad","log_delta_km","est_nuit",
                    "vitesse_km_min","montant_x_etranger","risque_x_nb_tx"]
       if f in df.columns]
CATS = ["type_transaction","device_type","segment_revenu","type_carte"]
df2 = df.copy()
for c in CATS:
    le = LabelEncoder()
    df2[c+"_enc"] = le.fit_transform(df2[c].astype(str))

FEATS = BASE + ENG + [c+"_enc" for c in CATS]
D     = len(FEATS)

X = df2[FEATS].values.astype(np.float32)
y = df2["fraude"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
sc = StandardScaler()
Xtr = sc.fit_transform(X_train).astype(np.float32)
Xte = sc.transform(X_test).astype(np.float32)

n_neg = int((y_train==0).sum())
n_pos = int((y_train==1).sum())
class_weight = {0: 1.0, 1: n_neg/n_pos}

# Légitimes uniquement pour l'Autoencoder
Xtr_legit = Xtr[y_train == 0]

print(f"   Features : {D}  (base={len(BASE)} + eng={len(ENG)} + cat={len(CATS)})")
print(f"   Train : {len(Xtr):,}  Test : {len(Xte):,}")
print(f"   class_weight fraud : {class_weight[1]:.1f}")

# Callbacks communs
def get_callbacks(name, patience=15):
    return [
        EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=0
        ),
    ]

def eval_model(yt, yp, thr=0.5):
    yc = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yc).ravel()
    return dict(
        auc_roc    = round(float(roc_auc_score(yt, yp)), 4),
        auc_pr     = round(float(average_precision_score(yt, yp)), 4),
        f1         = round(float(f1_score(yt, yc)), 4),
        precision  = round(float(precision_score(yt, yc, zero_division=0)), 4),
        recall     = round(float(recall_score(yt, yc)), 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

results = {}

# ─────────────────────────────────────────────────────────
# 2. DL-1 : MLP
# ─────────────────────────────────────────────────────────
print("\n[2/8] DL-1 : MLP (Dense 128→64→32→1)...")

def build_mlp(n_in):
    inp = Input(shape=(n_in,), name="input")
    x   = layers.Dense(128)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(64)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1,  activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="MLP_FraudDetector")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", curve="PR")]
    )
    return model

mlp = build_mlp(D)
mlp.summary(print_fn=lambda x: None)   # silencieux

t0 = time.time()
hist_mlp = mlp.fit(
    Xtr, y_train,
    epochs=200, batch_size=512,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=get_callbacks("mlp"),
    verbose=0
)
t_mlp = time.time() - t0
print(f"   Entraîné en {t_mlp:.1f}s ({len(hist_mlp.history['loss'])} epochs)")

yp_mlp = mlp.predict(Xte, verbose=0).ravel()
m_mlp  = eval_model(y_test, yp_mlp)
print(f"   AUC-PR={m_mlp['auc_pr']}  AUC-ROC={m_mlp['auc_roc']}  "
      f"F1={m_mlp['f1']}  FP={m_mlp['fp']}  FN={m_mlp['fn']}")

mlp.save("models/tf_mlp.keras")
m_mlp.update({
    "name":         "MLP (TensorFlow/Keras)",
    "architecture": f"Input({D})→128(BN+ReLU+DO)→64(BN+ReLU+DO)→32(ReLU)→1(Sigmoid)",
    "train_time_s": round(t_mlp, 1),
    "epochs":       len(hist_mlp.history["loss"]),
    "model_hash":   hashlib.sha256(open("models/tf_mlp.keras","rb").read()).hexdigest(),
})
results["mlp"] = m_mlp

# ─────────────────────────────────────────────────────────
# 3. DL-2 : AUTOENCODER
# ─────────────────────────────────────────────────────────
print("\n[3/8] DL-2 : Autoencoder (anomaly detection)...")
print("   Entraîné UNIQUEMENT sur transactions légitimes")

def build_autoencoder(n_in, encoding_dim=8):
    # Encoder
    enc_in  = Input(shape=(n_in,), name="enc_input")
    x       = layers.Dense(32, activation="relu")(enc_in)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(16, activation="relu")(x)
    encoded = layers.Dense(encoding_dim, activation="relu",
                            name="bottleneck")(x)
    encoder = Model(enc_in, encoded, name="Encoder")

    # Decoder
    dec_in  = Input(shape=(encoding_dim,), name="dec_input")
    x       = layers.Dense(16, activation="relu")(dec_in)
    x       = layers.Dense(32, activation="relu")(x)
    decoded = layers.Dense(n_in, activation="linear",
                            name="reconstruction")(x)
    decoder = Model(dec_in, decoded, name="Decoder")

    # Full autoencoder
    ae_in   = Input(shape=(n_in,))
    ae_out  = decoder(encoder(ae_in))
    ae      = Model(ae_in, ae_out, name="Autoencoder")
    ae.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")

    return ae, encoder, decoder

ae, encoder, decoder = build_autoencoder(D, encoding_dim=8)

t0 = time.time()
hist_ae = ae.fit(
    Xtr_legit, Xtr_legit,     # input = target (reconstruction)
    epochs=200, batch_size=256,
    validation_split=0.1,
    callbacks=get_callbacks("ae", patience=20),
    verbose=0
)
t_ae = time.time() - t0
print(f"   Entraîné en {t_ae:.1f}s ({len(hist_ae.history['loss'])} epochs)")

# Score = erreur MSE de reconstruction
recon = ae.predict(Xte, verbose=0)
err   = np.mean((Xte - recon)**2, axis=1)

# Seuil optimal = percentile 95 des légitimes du test
thr_ae = float(np.percentile(err[y_test == 0], 95))
yp_ae  = (err >= thr_ae).astype(int)
m_ae_roc = eval_model(y_test, err)
m_ae_cls = eval_model(y_test, yp_ae)
print(f"   Seuil={thr_ae:.5f}  AUC-PR={m_ae_roc['auc_pr']}  "
      f"AUC-ROC={m_ae_roc['auc_roc']}  FP={m_ae_cls['fp']}  FN={m_ae_cls['fn']}")

ae.save("models/tf_autoencoder.keras")
m_ae = dict(
    name                = "Autoencoder (TF — anomaly detection)",
    auc_pr              = m_ae_roc["auc_pr"],
    auc_roc             = m_ae_roc["auc_roc"],
    f1                  = m_ae_cls["f1"],
    precision           = m_ae_cls["precision"],
    recall              = m_ae_cls["recall"],
    tn=m_ae_cls["tn"], fp=m_ae_cls["fp"], fn=m_ae_cls["fn"], tp=m_ae_cls["tp"],
    reconstruction_threshold = round(thr_ae, 8),
    architecture        = "D→32(BN)→16→8(bottleneck) | 8→16→32→D",
    trained_on          = "légitimes uniquement",
    train_time_s        = round(t_ae, 1),
    epochs              = len(hist_ae.history["loss"]),
    model_hash          = hashlib.sha256(open("models/tf_autoencoder.keras","rb").read()).hexdigest(),
)
results["autoencoder"] = m_ae

# ─────────────────────────────────────────────────────────
# 4. DL-3 : MLP + ATTENTION
# ─────────────────────────────────────────────────────────
print("\n[4/8] DL-3 : MLP + Feature Attention...")

class FeatureAttention(layers.Layer):
    """
    Couche d'attention sur les features.
    Apprend un poids α_i ∈ [0,1] pour chaque feature i.
    X_out = X * softmax(Dense(tanh(Dense(X))))
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units, activation="tanh",  name="attn_hidden")
        self.W2 = layers.Dense(units, activation="linear", name="attn_logits")
        self.sm = layers.Softmax(axis=-1, name="attn_weights")

    def call(self, x, training=False):
        h     = self.W1(x)
        logits= self.W2(h)
        alpha = self.sm(logits)          # poids d'attention par feature
        return x * alpha, alpha          # features pondérées + poids

    def get_config(self):
        cfg = super().get_config()
        cfg["units"] = self.units
        return cfg

def build_attention_mlp(n_in):
    inp   = Input(shape=(n_in,), name="input")
    # Attention gate
    x_att, alpha = FeatureAttention(n_in, name="attention")(inp)
    # MLP sur features pondérées
    x = layers.Dense(128)(x_att)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    # Modèle principal (classification)
    model = Model(inp, out, name="AttentionMLP")
    model.compile(
        optimizer=keras.optimizers.Adam(8e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", curve="PR")]
    )

    # Modèle d'extraction des poids d'attention
    attn_extractor = Model(inp, alpha, name="AttentionExtractor")

    return model, attn_extractor

att_mlp, attn_extractor = build_attention_mlp(D)

t0 = time.time()
hist_att = att_mlp.fit(
    Xtr, y_train,
    epochs=200, batch_size=512,
    validation_split=0.15,
    class_weight=class_weight,
    callbacks=get_callbacks("att"),
    verbose=0
)
t_att = time.time() - t0
print(f"   Entraîné en {t_att:.1f}s ({len(hist_att.history['loss'])} epochs)")

yp_att = att_mlp.predict(Xte, verbose=0).ravel()
m_att  = eval_model(y_test, yp_att)

# Poids d'attention globaux
attn_w = attn_extractor.predict(Xte, verbose=0).mean(axis=0)
top5   = np.argsort(attn_w)[::-1][:5]
print(f"   AUC-PR={m_att['auc_pr']}  AUC-ROC={m_att['auc_roc']}  "
      f"F1={m_att['f1']}  FP={m_att['fp']}  FN={m_att['fn']}")
print("   Top 5 features (attention globale) :")
for i in top5:
    print(f"     {FEATS[i]:<28} : {attn_w[i]:.4f}")

att_mlp.save("models/tf_attention_mlp.keras")
m_att.update({
    "name":          "MLP + Feature Attention (TF/Keras)",
    "architecture":  "FeatureAttention(softmax) × Input → 128(BN+ReLU+DO)→64→32→1",
    "train_time_s":  round(t_att, 1),
    "epochs":        len(hist_att.history["loss"]),
    "model_hash":    hashlib.sha256(open("models/tf_attention_mlp.keras","rb").read()).hexdigest(),
    "top_attention": [
        {"rank": i+1, "feature": FEATS[top5[i]], "weight": round(float(attn_w[top5[i]]), 4)}
        for i in range(5)
    ],
})
results["attention_mlp"] = m_att

# ─────────────────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────────────────
print("\n[5/8] Figures...")

# ── Courbes d'apprentissage ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cfgs = [
    (hist_mlp, "MLP",           "#534AB7", "loss", "val_loss", "auc", "val_auc"),
    (hist_ae,  "Autoencoder",   "#993C1D", "loss", "val_loss", None,  None),
    (hist_att, "MLP+Attention", "#0F6E56", "loss", "val_loss", "auc", "val_auc"),
]
for ax, (hist, name, col, ktr, kvl, kauc, kvauc) in zip(axes, cfgs):
    H  = hist.history
    ep = range(1, len(H[ktr])+1)
    ax.plot(ep, H[ktr], color=col, alpha=0.35, lw=1.5, label="Train loss")
    ax.plot(ep, H[kvl], color=col, lw=2,       label="Val loss")
    if kauc and kvauc and kvauc in H:
        ax2 = ax.twinx()
        ax2.plot(ep, H[kvauc], color=col, lw=1.5, ls="--", alpha=0.7, label="Val AUC-PR")
        ax2.set_ylabel("AUC-PR", fontsize=8, color=col)
        ax2.tick_params(labelsize=7)
    ax.set_title(f"{name}  (ep={len(ep)})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Épochs", fontsize=9)
    ax.set_ylabel("Loss",   fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top"]].set_visible(False)
    ax.tick_params(labelsize=8)

plt.suptitle("Courbes d'apprentissage — TensorFlow/Keras",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/tf_01_learning_curves.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("   → reports/dl/tf_01_learning_curves.png")

# ── Comparaison ML vs DL ─────────────────────────────────
with open("reports/models_comparison.json") as f:
    mlr = json.load(f)

labs = ["RF\n(ML)", "GB/XGB\n(ML)", "LR\n(ML)",
        "MLP\n(TF)", "AE\n(TF)", "Att.\n(TF)"]
aucs = [mlr["models"]["random_forest"]["auc_pr"],
        mlr["models"]["gradient_boosting"]["auc_pr"],
        mlr["models"]["logistic_regression"]["auc_pr"],
        m_mlp["auc_pr"], m_ae["auc_pr"], m_att["auc_pr"]]
fps  = [mlr["models"]["random_forest"]["fp"],
        mlr["models"]["gradient_boosting"]["fp"],
        mlr["models"]["logistic_regression"]["fp"],
        m_mlp["fp"], m_ae["fp"], m_att["fp"]]
fns  = [mlr["models"]["random_forest"]["fn"],
        mlr["models"]["gradient_boosting"]["fn"],
        mlr["models"]["logistic_regression"]["fn"],
        m_mlp["fn"], m_ae["fn"], m_att["fn"]]
cols = ["#B4B2A9","#B4B2A9","#B4B2A9","#534AB7","#993C1D","#0F6E56"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
bars = ax.bar(range(6), aucs, color=cols, edgecolor="white",
              linewidth=0.5, width=0.65, alpha=0.88)
ax.set_xticks(range(6))
ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("AUC-PR", fontsize=10)
ax.set_ylim(0.70, 1.02)
ax.axhline(0.88, color="red", ls="--", lw=1, alpha=0.5, label="Seuil 0.88")
ax.axvline(2.5, color="gray", ls=":", lw=1, alpha=0.4)
ax.text(1.0, 1.005, "ML classique", ha="center", fontsize=9,
        color="#5F5E5A", fontweight="bold", transform=ax.get_xaxis_transform())
ax.text(4.5, 1.005, "TensorFlow/Keras", ha="center", fontsize=9,
        color="#534AB7", fontweight="bold", transform=ax.get_xaxis_transform())
for bar, v in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.004,
            f"{v:.4f}", ha="center", fontsize=8, fontweight="500")
ax.legend(fontsize=8)
ax.set_title("Comparaison AUC-PR — ML vs Deep Learning",
             fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
for lab, fp, fn, c in zip(labs, fps, fns, cols):
    ax.scatter(fp, fn, c=c, s=180, zorder=5, edgecolors="white", lw=1.5)
    ax.annotate(lab.replace("\n"," "), (fp, fn),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8.5, color=c)
ax.set_xlabel("Faux Positifs (clients bloqués à tort)", fontsize=10)
ax.set_ylabel("Faux Négatifs (fraudes manquées)",       fontsize=10)
ax.set_title("Trade-off FP / FN\n(idéal = coin bas-gauche)",
             fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

plt.suptitle("ML vs TensorFlow/Keras — Synthèse comparative",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/tf_02_ml_vs_dl.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("   → reports/dl/tf_02_ml_vs_dl.png")

# ── Attention + Autoencoder ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
si  = np.argsort(attn_w)
sv  = attn_w[si]
sf  = [FEATS[i] for i in si]
cc  = plt.cm.RdYlGn(sv / sv.max())
ax.barh(range(len(sf)), sv, color=cc, edgecolor="white",
        linewidth=0.4, alpha=0.87)
ax.set_yticks(range(len(sf)))
ax.set_yticklabels(sf, fontsize=8)
ax.set_xlabel("Poids d'attention moyen", fontsize=9)
ax.set_title("MLP+Attention — Importance globale des features\n"
             "(interprétabilité native, sans SHAP)",
             fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
el = err[y_test==0]
ef = err[y_test==1]
clip_max = float(np.percentile(err, 99))
ax.hist(el, bins=60, density=True, color="#1D9E75", alpha=0.6,
        label="Légitimes", range=(0, clip_max))
ax.hist(ef, bins=60, density=True, color="#E24B4A", alpha=0.75,
        label="Fraudes",   range=(0, clip_max))
ax.axvline(thr_ae, color="black", ls="--", lw=1.5,
           label=f"Seuil ({thr_ae:.4f})")
ax.set_xlabel("Erreur de reconstruction MSE", fontsize=9)
ax.set_ylabel("Densité",                      fontsize=9)
ax.set_title("Autoencoder — Distribution des erreurs\n"
             "Fraudes = reconstruction difficile → score élevé",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, frameon=False)
ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Interprétabilité des modèles TensorFlow/Keras",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/tf_03_interpretability.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("   → reports/dl/tf_03_interpretability.png")

# ─────────────────────────────────────────────────────────
# 6. RAPPORT JSON (format blockchain)
# ─────────────────────────────────────────────────────────
print("\n[6/8] Rapport JSON...")

report = {
    "experiment":   "fraud_detection_tensorflow_keras",
    "run_date":     datetime.now().isoformat(),
    "framework":    f"TensorFlow {tf.__version__} / Keras",
    "n_features":   D,
    "dataset":      DATA,
    "ml_models":    mlr["models"],
    "dl_models":    results,
    "best_dl": max(results, key=lambda k: results[k]["auc_pr"]),
    "autoencoder_threshold": round(thr_ae, 8),
    "deployment": {
        "primary":   "XGBoost (ML) — production temps réel",
        "anomaly":   "Autoencoder (TF) — détection patterns inconnus",
        "interpret": "MLP+Attention (TF) — explicabilité par feature",
        "blockchain":"hash SHA-256 → RegisterModel() Hyperledger Fabric"
    }
}
with open("reports/dl/tf_full_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────
# 7. INFÉRENCE EXEMPLE (format blockchain RecordDecision)
# ─────────────────────────────────────────────────────────
print("\n[7/8] Exemple d'inférence avec payload blockchain...")

fraud_idx = np.where(y_test==1)[0]
sample_tx  = Xte[fraud_idx[0]:fraud_idx[0]+1]
score_mlp  = float(mlp.predict(sample_tx, verbose=0)[0][0])
score_att  = float(att_mlp.predict(sample_tx, verbose=0)[0][0])
score_ae   = float(ae.predict(sample_tx, verbose=0).mean())
attn_sample= attn_extractor.predict(sample_tx, verbose=0)[0]
top3_attn  = [{"feature":FEATS[i],"weight":round(float(attn_sample[i]),4)}
               for i in np.argsort(attn_sample)[::-1][:3]]

blockchain_payload = {
    "tx_id":          f"TX_DEMO_{fraud_idx[0]:08d}",
    "model_version":  "tf_attention_mlp_v1.0",
    "score_fraude":   round(score_att, 4),
    "zone":           "RED" if score_att>0.85 else "AMBER" if score_att>0.40 else "GREEN",
    "decision":       "FRAUDE" if score_att>0.85 else "REVUE_HUMAINE",
    "top_features":   top3_attn,
    "scores_all_models": {"mlp":round(score_mlp,4),"attention_mlp":round(score_att,4)},
    "model_hash":     m_att["model_hash"][:16]+"...",
    "explication_hash":hashlib.sha256(json.dumps(top3_attn,sort_keys=True).encode()).hexdigest(),
    "framework":      "TensorFlow/Keras"
}
print(f"   Score fraude : {score_att:.4f}  →  {blockchain_payload['zone']}")
print(f"   Top feature  : {top3_attn[0]['feature']} ({top3_attn[0]['weight']:.4f})")

with open("reports/dl/tf_blockchain_payload_example.json","w") as f:
    json.dump(blockchain_payload, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────
# 8. RÉSUMÉ
# ─────────────────────────────────────────────────────────
print("\n[8/8] Résumé final")
print("\n" + "="*68)
print("  RÉSUMÉ — ML + TENSORFLOW/KERAS")
print("="*68)
print(f"  {'Modèle':<36} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>7} {'FP':>5} {'FN':>5}")
print("  "+"-"*66)
rows = [
    ("Random Forest (ML)",          mlr["models"]["random_forest"]),
    ("XGBoost / Grad.Boost (ML)",   mlr["models"]["gradient_boosting"]),
    ("Logistic Regression (ML)",    mlr["models"]["logistic_regression"]),
    ("MLP (TensorFlow/Keras)",      m_mlp),
    ("Autoencoder (TF — anomalie)", m_ae),
    ("MLP + Attention (TF/Keras)",  m_att),
]
best_pr = max(r["auc_pr"] for _, r in rows)
for name, r in rows:
    star = " ★" if abs(r["auc_pr"]-best_pr) < 1e-4 else ""
    print(f"  {name:<36} {r['auc_pr']:>8.4f} {r['auc_roc']:>8.4f} "
          f"{r['f1']:>7.4f} {r['fp']:>5} {r['fn']:>5}{star}")

print(f"\n  ★ = Meilleur AUC-PR")
print(f"\n  Modèles Keras sauvegardés :")
for p in ["models/tf_mlp.keras",
          "models/tf_autoencoder.keras",
          "models/tf_attention_mlp.keras"]:
    print(f"    {p}")
print(f"\n  Figures :")
for p in ["reports/dl/tf_01_learning_curves.png",
          "reports/dl/tf_02_ml_vs_dl.png",
          "reports/dl/tf_03_interpretability.png"]:
    print(f"    {p}")
print(f"\n  Rapport  : reports/dl/tf_full_report.json")
print(f"  Blockchain: reports/dl/tf_blockchain_payload_example.json")
print("="*68)
