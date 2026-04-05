"""
=============================================================
DEEP LEARNING — PYTORCH
Gouvernance et Traçabilité des Modèles IA — Fraude Bancaire
=============================================================
3 modèles avec PyTorch :

DL-1 : MLP  → réseau dense classique
DL-2 : Autoencoder → détection d'anomalies (légitimes only)
DL-3 : MLP + Attention → interprétabilité native par feature

Installation :
    pip install torch torchvision     # CPU
    pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

Compatible : PyTorch 2.0+, Python 3.9+
=============================================================
"""

import numpy as np
import pandas as pd
import json, pickle, hashlib, os, warnings, time
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models",     exist_ok=True)
os.makedirs("reports/dl", exist_ok=True)

print("=" * 65)
print(f"  PYTORCH — v{torch.__version__}")
print(f"  Device : {DEVICE}")
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
pos_weight = torch.tensor([n_neg/n_pos], dtype=torch.float32).to(DEVICE)

Xtr_legit = Xtr[y_train == 0]

# PyTorch tensors
def to_tensor(X, y=None):
    Xt = torch.FloatTensor(X).to(DEVICE)
    if y is not None:
        yt = torch.FloatTensor(y).to(DEVICE)
        return TensorDataset(Xt, yt)
    return Xt

ds_train    = to_tensor(Xtr, y_train)
ds_train_l  = to_tensor(Xtr_legit, Xtr_legit)  # autoencoder
dl_train    = DataLoader(ds_train,   batch_size=512,  shuffle=True,  drop_last=False)
dl_legit    = DataLoader(ds_train_l, batch_size=256,  shuffle=True,  drop_last=False)

print(f"   Features : {D}  |  Train : {len(Xtr):,}  |  Test : {len(Xte):,}")
print(f"   pos_weight (fraude) : {pos_weight.item():.1f}")

def eval_model(yt, yp, thr=0.5):
    yc = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yc).ravel()
    return dict(
        auc_roc   = round(float(roc_auc_score(yt, yp)), 4),
        auc_pr    = round(float(average_precision_score(yt, yp)), 4),
        f1        = round(float(f1_score(yt, yc)), 4),
        precision = round(float(precision_score(yt, yc, zero_division=0)), 4),
        recall    = round(float(recall_score(yt, yc)), 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

results = {}

# ─────────────────────────────────────────────────────────
# 2. DL-1 : MLP
# ─────────────────────────────────────────────────────────
print("\n[2/8] DL-1 : MLP (Dense 128→64→32→1)...")

class MLP(nn.Module):
    def __init__(self, n_in, hidden=[128,64,32], dropout=0.3):
        super().__init__()
        layers_list = []
        in_dim = n_in
        for h in hidden:
            layers_list += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers_list += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_classifier(model, dl, Xval_t, yval_t, loss_fn,
                     epochs=200, patience=15, lr=1e-3):
    opt  = optim.Adam(model.parameters(), lr=lr)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor=0.5)
    best_val, wait, hist = float("inf"), 0, {"tl":[],"vl":[],"va":[]}

    for ep in range(epochs):
        model.train()
        tl = 0.0; nb = 0
        for Xb, yb in dl:
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        tl /= nb
        model.eval()
        with torch.no_grad():
            vp  = model(Xval_t).cpu().numpy()
            vl  = loss_fn(model(Xval_t), yval_t).item()
            va  = roc_auc_score(yval_t.cpu().numpy(), vp)
        sch.step(vl)
        hist["tl"].append(tl); hist["vl"].append(vl); hist["va"].append(va)
        if vl < best_val - 1e-5:
            best_val = vl; wait = 0
            torch.save(model.state_dict(), "models/pt_best_tmp.pt")
        else:
            wait += 1
            if wait >= patience:
                model.load_state_dict(torch.load("models/pt_best_tmp.pt",
                                                  weights_only=True))
                break
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"     Ep{ep+1:3d} tr={tl:.4f} vl={vl:.4f} AUC={va:.4f}")
    print(f"     Stop ep{ep+1}  best_val={best_val:.4f}")
    return hist

Xval_t = torch.FloatTensor(Xtr[-6000:]).to(DEVICE)
yval_t = torch.FloatTensor(y_train[-6000:]).to(DEVICE)

mlp   = MLP(D, [128,64,32], dropout=0.3).to(DEVICE)
bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Adapter loss pour sortie sigmoid déjà appliquée
bce_sig = nn.BCELoss()

t0 = time.time()
hist_mlp = train_classifier(mlp, dl_train, Xval_t, yval_t, bce_sig,
                             epochs=200, patience=15, lr=1e-3)
t_mlp = time.time() - t0
print(f"   Entraîné en {t_mlp:.1f}s")

mlp.eval()
with torch.no_grad():
    yp_mlp = mlp(torch.FloatTensor(Xte).to(DEVICE)).cpu().numpy()
m_mlp = eval_model(y_test, yp_mlp)
print(f"   AUC-PR={m_mlp['auc_pr']}  AUC-ROC={m_mlp['auc_roc']}  "
      f"F1={m_mlp['f1']}  FP={m_mlp['fp']}  FN={m_mlp['fn']}")

torch.save(mlp.state_dict(), "models/pt_mlp.pt")
with open("models/pt_mlp_arch.json","w") as f:
    json.dump({"n_in":D,"hidden":[128,64,32],"dropout":0.3}, f)
m_mlp.update({
    "name":         "MLP (PyTorch)",
    "architecture": f"Input({D})→128(BN+ReLU+DO)→64(BN+ReLU+DO)→32(ReLU)→1(Sigmoid)",
    "train_time_s": round(t_mlp, 1),
    "epochs":       len(hist_mlp["tl"]),
    "model_hash":   hashlib.sha256(open("models/pt_mlp.pt","rb").read()).hexdigest(),
})
results["mlp"] = m_mlp

# ─────────────────────────────────────────────────────────
# 3. DL-2 : AUTOENCODER
# ─────────────────────────────────────────────────────────
print("\n[3/8] DL-2 : Autoencoder (anomaly detection)...")
print("   Entraîné UNIQUEMENT sur légitimes")

class Autoencoder(nn.Module):
    def __init__(self, n_in, enc_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16),   nn.ReLU(),
            nn.Linear(16, enc_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(enc_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),      nn.ReLU(),
            nn.Linear(32, n_in),    # sortie linéaire
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(model, dl, Xval_legit_t, epochs=200, patience=15, lr=5e-4):
    opt  = optim.Adam(model.parameters(), lr=lr)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    mse  = nn.MSELoss()
    best_val, wait, hist = float("inf"), 0, {"tl":[],"vl":[]}

    for ep in range(epochs):
        model.train()
        tl = 0.0; nb = 0
        for Xb, _ in dl:
            opt.zero_grad()
            loss = mse(model(Xb), Xb)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        tl /= nb
        model.eval()
        with torch.no_grad():
            vl = mse(model(Xval_legit_t), Xval_legit_t).item()
        sch.step(vl)
        hist["tl"].append(tl); hist["vl"].append(vl)
        if vl < best_val - 1e-7:
            best_val = vl; wait = 0
            torch.save(model.state_dict(), "models/pt_ae_best_tmp.pt")
        else:
            wait += 1
            if wait >= patience:
                model.load_state_dict(torch.load("models/pt_ae_best_tmp.pt",
                                                  weights_only=True))
                break
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"     Ep{ep+1:3d} MSE_tr={tl:.5f} MSE_vl={vl:.5f}")
    print(f"     Stop ep{ep+1}  best_MSE={best_val:.6f}")
    return hist

idx_l = np.where(y_train==0)[0]; np.random.shuffle(idx_l)
sp    = int(len(idx_l)*0.1)
Xae_val_t = torch.FloatTensor(Xtr[idx_l[:sp]]).to(DEVICE)
ds_legit_ae = TensorDataset(torch.FloatTensor(Xtr[idx_l[sp:]]).to(DEVICE),
                              torch.zeros(len(idx_l)-sp))
dl_ae = DataLoader(ds_legit_ae, batch_size=256, shuffle=True)

ae   = Autoencoder(D, enc_dim=8).to(DEVICE)
t0   = time.time()
hist_ae = train_autoencoder(ae, dl_ae, Xae_val_t, epochs=200, patience=15)
t_ae = time.time() - t0
print(f"   Entraîné en {t_ae:.1f}s")

ae.eval()
Xte_t = torch.FloatTensor(Xte).to(DEVICE)
with torch.no_grad():
    recon = ae(Xte_t).cpu().numpy()
err   = np.mean((Xte - recon)**2, axis=1)
thr_ae= float(np.percentile(err[y_test==0], 95))
yp_ae = (err >= thr_ae).astype(int)
m_ae_roc = eval_model(y_test, err)
m_ae_cls = eval_model(y_test, yp_ae)
print(f"   Seuil={thr_ae:.5f}  AUC-PR={m_ae_roc['auc_pr']}  "
      f"AUC-ROC={m_ae_roc['auc_roc']}  FP={m_ae_cls['fp']}  FN={m_ae_cls['fn']}")

torch.save(ae.state_dict(), "models/pt_autoencoder.pt")
m_ae = dict(
    name="Autoencoder (PyTorch — anomaly detection)",
    auc_pr=m_ae_roc["auc_pr"], auc_roc=m_ae_roc["auc_roc"],
    f1=m_ae_cls["f1"], precision=m_ae_cls["precision"], recall=m_ae_cls["recall"],
    tn=m_ae_cls["tn"], fp=m_ae_cls["fp"], fn=m_ae_cls["fn"], tp=m_ae_cls["tp"],
    reconstruction_threshold=round(thr_ae, 8),
    architecture="D→32(BN)→16→8(bottleneck) | 8→16→32→D",
    trained_on="légitimes uniquement",
    train_time_s=round(t_ae, 1), epochs=len(hist_ae["tl"]),
    model_hash=hashlib.sha256(open("models/pt_autoencoder.pt","rb").read()).hexdigest(),
)
results["autoencoder"] = m_ae

# ─────────────────────────────────────────────────────────
# 4. DL-3 : MLP + ATTENTION
# ─────────────────────────────────────────────────────────
print("\n[4/8] DL-3 : MLP + Feature Attention...")

class FeatureAttention(nn.Module):
    """Poids d'attention par feature — interprétabilité native."""
    def __init__(self, n_in):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_in)
        self.fc2 = nn.Linear(n_in, n_in)

    def forward(self, x):
        h     = torch.tanh(self.fc1(x))
        logits= self.fc2(h)
        alpha = torch.softmax(logits, dim=-1)
        return x * alpha, alpha

class AttentionMLP(nn.Module):
    def __init__(self, n_in, hidden=[128,64,32], dropout=0.3):
        super().__init__()
        self.attention = FeatureAttention(n_in)
        layers_list    = []
        in_dim = n_in
        for h in hidden:
            layers_list += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers_list += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers_list)

    def forward(self, x):
        x_att, alpha = self.attention(x)
        return self.mlp(x_att).squeeze(-1), alpha

    def predict(self, x):
        out, _ = self.forward(x)
        return out

    def get_attention(self, x):
        _, alpha = self.forward(x)
        return alpha

att_mlp = AttentionMLP(D, [128,64,32], dropout=0.3).to(DEVICE)

# Wrapper pour train_classifier
class AttWrapper(nn.Module):
    def __init__(self, model): super().__init__(); self.m=model
    def forward(self, x): return self.m.predict(x)

t0 = time.time()
hist_att = train_classifier(
    AttWrapper(att_mlp), dl_train, Xval_t, yval_t, bce_sig,
    epochs=200, patience=15, lr=8e-4
)
t_att = time.time() - t0
print(f"   Entraîné en {t_att:.1f}s")

att_mlp.eval()
with torch.no_grad():
    yp_att   = att_mlp.predict(Xte_t).cpu().numpy()
    attn_w   = att_mlp.get_attention(Xte_t).cpu().numpy().mean(axis=0)
m_att = eval_model(y_test, yp_att)
top5  = np.argsort(attn_w)[::-1][:5]
print(f"   AUC-PR={m_att['auc_pr']}  AUC-ROC={m_att['auc_roc']}  "
      f"F1={m_att['f1']}  FP={m_att['fp']}  FN={m_att['fn']}")
print("   Top 5 features (attention) :")
for i in top5:
    print(f"     {FEATS[i]:<28} : {attn_w[i]:.4f}")

torch.save(att_mlp.state_dict(), "models/pt_attention_mlp.pt")
m_att.update({
    "name":          "MLP + Feature Attention (PyTorch)",
    "architecture":  "FeatureAttention(softmax)×Input → 128(BN+ReLU+DO)→64→32→1",
    "train_time_s":  round(t_att, 1),
    "epochs":        len(hist_att["tl"]),
    "model_hash":    hashlib.sha256(open("models/pt_attention_mlp.pt","rb").read()).hexdigest(),
    "top_attention": [
        {"rank":i+1,"feature":FEATS[top5[i]],"weight":round(float(attn_w[top5[i]]),4)}
        for i in range(5)
    ],
})
results["attention_mlp"] = m_att

# ─────────────────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────────────────
print("\n[5/8] Figures...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cfgs = [
    (hist_mlp, "MLP",           "#534AB7", "tl","vl","va"),
    (hist_ae,  "Autoencoder",   "#993C1D", "tl","vl",None),
    (hist_att, "MLP+Attention", "#0F6E56", "tl","vl","va"),
]
for ax, (H, name, col, ktr, kvl, kva) in zip(axes, cfgs):
    ep = range(1, len(H[ktr])+1)
    ax.plot(ep, H[ktr], color=col, alpha=0.35, lw=1.5, label="Train loss")
    ax.plot(ep, H[kvl], color=col, lw=2,       label="Val loss")
    if kva and kva in H:
        ax2 = ax.twinx()
        ax2.plot(ep, H[kva], color=col, lw=1.5, ls="--", alpha=0.7, label="Val AUC")
        ax2.set_ylabel("AUC", fontsize=8, color=col); ax2.tick_params(labelsize=7)
    ax.set_title(f"{name}  (ep={len(ep)})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Épochs", fontsize=9); ax.set_ylabel("Loss", fontsize=9)
    ax.legend(fontsize=8); ax.spines[["top"]].set_visible(False)
plt.suptitle("Courbes d'apprentissage — PyTorch", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/pt_01_learning_curves.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()

with open("reports/models_comparison.json") as f: mlr=json.load(f)
labs=["RF\n(ML)","GB/XGB\n(ML)","LR\n(ML)","MLP\n(PT)","AE\n(PT)","Att.\n(PT)"]
aucs=[mlr["models"]["random_forest"]["auc_pr"],mlr["models"]["gradient_boosting"]["auc_pr"],
      mlr["models"]["logistic_regression"]["auc_pr"],m_mlp["auc_pr"],m_ae["auc_pr"],m_att["auc_pr"]]
fps =[mlr["models"]["random_forest"]["fp"],mlr["models"]["gradient_boosting"]["fp"],
      mlr["models"]["logistic_regression"]["fp"],m_mlp["fp"],m_ae["fp"],m_att["fp"]]
fns =[mlr["models"]["random_forest"]["fn"],mlr["models"]["gradient_boosting"]["fn"],
      mlr["models"]["logistic_regression"]["fn"],m_mlp["fn"],m_ae["fn"],m_att["fn"]]
cols=["#B4B2A9","#B4B2A9","#B4B2A9","#534AB7","#993C1D","#0F6E56"]

fig,axes=plt.subplots(1,2,figsize=(16,6))
ax=axes[0]
bars=ax.bar(range(6),aucs,color=cols,edgecolor="white",lw=0.5,width=0.65,alpha=0.88)
ax.set_xticks(range(6)); ax.set_xticklabels(labs,fontsize=9)
ax.set_ylabel("AUC-PR",fontsize=10); ax.set_ylim(0.70,1.02)
ax.axhline(0.88,color="red",ls="--",lw=1,alpha=0.5,label="Seuil 0.88")
ax.axvline(2.5,color="gray",ls=":",lw=1,alpha=0.4)
ax.text(1.0,1.005,"ML classique",ha="center",fontsize=9,color="#5F5E5A",fontweight="bold",transform=ax.get_xaxis_transform())
ax.text(4.5,1.005,"PyTorch",ha="center",fontsize=9,color="#534AB7",fontweight="bold",transform=ax.get_xaxis_transform())
for bar,v in zip(bars,aucs): ax.text(bar.get_x()+bar.get_width()/2,v+0.004,f"{v:.4f}",ha="center",fontsize=8)
ax.legend(fontsize=8); ax.set_title("Comparaison AUC-PR — ML vs PyTorch",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
ax=axes[1]
for lab,fp,fn,c in zip(labs,fps,fns,cols):
    ax.scatter(fp,fn,c=c,s=180,zorder=5,edgecolors="white",lw=1.5)
    ax.annotate(lab.replace("\n"," "),(fp,fn),xytext=(6,4),textcoords="offset points",fontsize=8.5,color=c)
ax.set_xlabel("Faux Positifs",fontsize=10); ax.set_ylabel("Faux Négatifs",fontsize=10)
ax.set_title("Trade-off FP/FN\n(idéal=coin bas-gauche)",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
plt.suptitle("ML vs PyTorch — Synthèse",fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/pt_02_ml_vs_dl.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

fig,axes=plt.subplots(1,2,figsize=(16,6))
ax=axes[0]; si=np.argsort(attn_w); sv=attn_w[si]; sf=[FEATS[i] for i in si]
cc=plt.cm.RdYlGn(sv/sv.max())
ax.barh(range(len(sf)),sv,color=cc,edgecolor="white",lw=0.4,alpha=0.87)
ax.set_yticks(range(len(sf))); ax.set_yticklabels(sf,fontsize=8)
ax.set_xlabel("Poids d'attention moyen",fontsize=9)
ax.set_title("MLP+Attention — Importance des features\n(interprétabilité sans SHAP)",fontsize=11,fontweight="bold"); ax.spines[["top","right"]].set_visible(False)
ax=axes[1]; el=err[y_test==0]; ef=err[y_test==1]; clip_m=float(np.percentile(err,99))
ax.hist(el,bins=60,density=True,color="#1D9E75",alpha=0.6,label="Légitimes",range=(0,clip_m))
ax.hist(ef,bins=60,density=True,color="#E24B4A",alpha=0.75,label="Fraudes",range=(0,clip_m))
ax.axvline(thr_ae,color="black",ls="--",lw=1.5,label=f"Seuil ({thr_ae:.4f})")
ax.set_xlabel("Erreur reconstruction MSE",fontsize=9); ax.set_ylabel("Densité",fontsize=9)
ax.set_title("Autoencoder — Distribution des erreurs\nFraudes = mal reconstruites",fontsize=11,fontweight="bold")
ax.legend(fontsize=9,frameon=False); ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Interprétabilité — PyTorch",fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig("reports/dl/pt_03_interpretability.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
print("   Figures OK")

# ─────────────────────────────────────────────────────────
# 6. RAPPORT
# ─────────────────────────────────────────────────────────
print("\n[6/8] Rapport JSON...")
report={
    "experiment":"fraud_detection_pytorch","run_date":datetime.now().isoformat(),
    "framework":f"PyTorch {torch.__version__}","device":str(DEVICE),
    "n_features":D,"dataset":DATA,"ml_models":mlr["models"],"dl_models":results,
    "best_dl":max(results,key=lambda k:results[k]["auc_pr"]),
    "autoencoder_threshold":round(thr_ae,8),
    "deployment":{"primary":"XGBoost (ML)","anomaly":"Autoencoder (PT)","explain":"MLP+Attention (PT)"}
}
with open("reports/dl/pt_full_report.json","w",encoding="utf-8") as f:
    json.dump(report,f,indent=2,ensure_ascii=False)

# ─────────────────────────────────────────────────────────
# RÉSUMÉ
# ─────────────────────────────────────────────────────────
print("\n[8/8] Résumé")
print("\n"+"="*68); print("  RÉSUMÉ — ML + PYTORCH"); print("="*68)
print(f"  {'Modèle':<36} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>7} {'FP':>5} {'FN':>5}")
print("  "+"-"*66)
rows=[("Random Forest (ML)",mlr["models"]["random_forest"]),
      ("XGBoost/GradBoost (ML)",mlr["models"]["gradient_boosting"]),
      ("Logistic Regression (ML)",mlr["models"]["logistic_regression"]),
      ("MLP (PyTorch)",m_mlp),("Autoencoder (PyTorch)",m_ae),("MLP+Attention (PyTorch)",m_att)]
best_pr=max(r["auc_pr"] for _,r in rows)
for name,r in rows:
    s=" ★" if abs(r["auc_pr"]-best_pr)<1e-4 else ""
    print(f"  {name:<36} {r['auc_pr']:>8.4f} {r['auc_roc']:>8.4f} {r['f1']:>7.4f} {r['fp']:>5} {r['fn']:>5}{s}")
print(f"\n  ★ = Meilleur AUC-PR  |  Device utilisé : {DEVICE}")
for p in ["models/pt_mlp.pt","models/pt_autoencoder.pt","models/pt_attention_mlp.pt",
          "reports/dl/pt_01_learning_curves.png","reports/dl/pt_02_ml_vs_dl.png",
          "reports/dl/pt_03_interpretability.png","reports/dl/pt_full_report.json"]:
    print(f"  {p}")
print("="*68)
