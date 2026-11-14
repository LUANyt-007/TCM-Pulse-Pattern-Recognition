import os
import copy
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    accuracy_score, hamming_loss, precision_score, recall_score
)
from skmultilearn.model_selection import IterativeStratification
import matplotlib.pyplot as plt

# ===================== Configuration =====================
# Chinese names are kept as dictionary keys because they are used to parse
# pulse labels directly from the sample file names.
PULSE_DICT = {
    "细脉": 0, "滑脉": 1, "短脉": 2, "弦脉": 3, "沉脉": 4,
    "虚脉": 5, "浮脉": 6, "洪脉": 7, "实脉": 8, "涩脉": 9,
    "数脉": 10, "迟脉": 11, "促脉": 12, "紧脉": 13, "缓脉": 14,
}
# English display names, used only for figures and printed summaries.
PULSE_EN = {
    "细脉": "Thready", "滑脉": "Slippery", "短脉": "Short", "弦脉": "Wiry",
    "沉脉": "Deep", "虚脉": "Deficient", "浮脉": "Floating", "洪脉": "Surging",
    "实脉": "Excess", "涩脉": "Rough", "数脉": "Rapid", "迟脉": "Slow",
    "促脉": "Rapid-irregular", "紧脉": "Tense", "缓脉": "Moderate",
}
CLASS_NUM = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/Users/luan/Desktop/"
REAL_DIR = "/Users/luan/Desktop/"  # reserved for the separate external-test script
OUT_DIR = os.path.join(os.path.dirname(DATA_DIR), "pulse_eval_results_v2")

K = 5
REPEATS = 1       # Start with 1 repeat on CPU; increase only if runtime allows.
EPOCHS = 30
PATIENCE = 6      # Stop after this many epochs without validation Macro-F1 improvement.
MIN_DELTA = 1e-4
BATCH = 16
LR = 0.001
NUM_WORKERS = 0   # 0 is a safe default on macOS.
SEED = 42
THRESHOLD = 0.5
SEQ_LEN = 6000    # Length of one pulse recording (used by the BP baseline).

name_to_id = dict(PULSE_DICT)
CLASS_NAMES = [PULSE_EN[name] for name, _ in sorted(PULSE_DICT.items(), key=lambda kv: kv[1])]
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================== Label graphs =====================
def build_graphs():
    A_pos = torch.eye(CLASS_NUM)
    relations = [
        ("浮脉", "紧脉"), ("浮脉", "数脉"), ("浮脉", "滑脉"), ("浮脉", "弦脉"),
        ("浮脉", "涩脉"), ("浮脉", "洪脉"), ("浮脉", "细脉"),
        ("沉脉", "迟脉"), ("沉脉", "细脉"), ("沉脉", "弦脉"), ("沉脉", "滑脉"),
        ("沉脉", "涩脉"), ("沉脉", "数脉"),
        ("弦脉", "数脉"), ("弦脉", "滑脉"), ("弦脉", "细脉"), ("弦脉", "迟脉"),
        ("滑脉", "数脉"), ("细脉", "数脉"), ("洪脉", "数脉"), ("虚脉", "数脉"),
        ("虚脉", "细脉"),
    ]
    for p1, p2 in relations:
        i, j = name_to_id[p1], name_to_id[p2]
        A_pos[i, j] = A_pos[j, i] = 1

    A_neg = torch.zeros(CLASS_NUM, CLASS_NUM)
    exclusive_relations = [
        ("迟脉", "数脉"), ("浮脉", "沉脉"),
        ("虚脉", "实脉"), ("滑脉", "涩脉"),
    ]
    for p1, p2 in exclusive_relations:
        i, j = name_to_id[p1], name_to_id[p2]
        A_neg[i, j] = A_neg[j, i] = 1

    def norm(A):
        degree = torch.diag(1.0 / torch.sqrt(A.sum(dim=1) + 1e-8))
        return degree @ A @ degree

    return norm(A_pos).to(DEVICE), norm(A_neg).to(DEVICE)


A_POS_NORM, A_NEG_NORM = build_graphs()
A_NEG_ZERO = torch.zeros_like(A_NEG_NORM)


# ===================== Data loading =====================
def load_all(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    X, y, files = [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npy") or fname.endswith("_label.npy"):
            continue
        path = os.path.join(data_dir, fname)
        d = np.load(path).astype(np.float32)
        if d.ndim != 2:
            print(f"[Skip] {fname}: expected 2-D array, got shape {d.shape}")
            continue
        # Per-file, per-channel normalization as in the original script.
        d = (d - d.mean(axis=0, keepdims=True)) / (d.std(axis=0, keepdims=True) + 1e-8)
        d = d.T  # expected (4, sequence_length)
        if d.shape[0] != 4:
            print(f"[Skip] {fname}: expected 4 channels after transpose, got {d.shape}")
            continue

        label = np.zeros(CLASS_NUM, dtype=np.float32)
        for pulse_name, idx in PULSE_DICT.items():
            if pulse_name in fname:
                label[idx] = 1.0
        if label.sum() == 0:
            print(f"[Warning] No pulse label detected in filename: {fname}")
        X.append(d)
        y.append(label)
        files.append(fname)

    if not X:
        raise ValueError(f"No valid .npy samples found in {data_dir}")
    # Sequence length must be consistent for stacking.
    lengths = {x.shape[1] for x in X}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent sequence lengths found: {sorted(lengths)}")
    return np.stack(X), np.stack(y), np.asarray(files)


# ===================== Model components =====================
class PAAF(nn.Module):
    """Pulse-Adaptive Activation Function: the negative slope is generated
    from the channel-wise global statistics instead of being fixed."""

    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.alpha_gen = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x):
        _, c, _ = x.shape
        z = self.gap(x).squeeze(-1)
        alpha = self.alpha_gen(z).view(-1, c, 1)
        return torch.where(x > 0, x, alpha * x)


class MultiScaleConv(nn.Module):
    """Multi-scale 1D convolution (kernels 3/5/7/9) with a selectable activation."""

    def __init__(self, ic, oc, activation="paaf"):
        super().__init__()
        self.c3 = nn.Conv1d(ic, oc, 3, padding=1)
        self.c5 = nn.Conv1d(ic, oc, 5, padding=2)
        self.c7 = nn.Conv1d(ic, oc, 7, padding=3)
        self.c9 = nn.Conv1d(ic, oc, 9, padding=4)
        self.act = PAAF(oc * 4) if activation == "paaf" else nn.ReLU()

    def forward(self, x):
        x = torch.cat([self.c3(x), self.c5(x), self.c7(x), self.c9(x)], dim=1)
        return self.act(x)


class CNNFeature(nn.Module):
    """Multi-scale 1D-CNN backbone with dual-channel (avg + max) pooling."""

    def __init__(self, use_paaf=True):
        super().__init__()
        act = "paaf" if use_paaf else "relu"
        self.l1 = nn.Sequential(MultiScaleConv(4, 32, act), nn.MaxPool1d(4))
        self.l2 = nn.Sequential(MultiScaleConv(128, 64, act), nn.MaxPool1d(4))
        self.l3 = nn.Sequential(MultiScaleConv(256, 128, act), nn.MaxPool1d(4))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.maxp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        a = self.avg(x).squeeze(-1)
        m = self.maxp(x).squeeze(-1)
        return torch.cat([a, m], dim=1)


class LightweightGCN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc_pos = nn.Linear(d, d)
        self.fc_neg = nn.Linear(d, d)
        self.rho = nn.Parameter(torch.tensor(-2.0))

    def forward(self, H, A_pos, A_neg):
        lam = F.softplus(self.rho)
        Hp = self.fc_pos(torch.einsum("ij,bjf->bif", A_pos, H))
        Hn = self.fc_neg(torch.einsum("ij,bjf->bif", A_neg, H))
        return torch.relu(Hp - lam * Hn)


class PulseGCN(nn.Module):
    """Full model: multiscale CNN + PAAF + positive/negative graph convolution."""

    def __init__(self, use_paaf=True):
        super().__init__()
        self.cnn = CNNFeature(use_paaf)
        self.node_fc = nn.Linear(1024, CLASS_NUM * 64)
        self.node_emb = nn.Parameter(torch.randn(CLASS_NUM, 64))
        self.gcn = LightweightGCN(64)
        self.out = nn.Linear(64, 1)

    def forward(self, x, A_pos, A_neg):
        node = self.node_fc(self.cnn(x)).view(-1, CLASS_NUM, 64)
        node = node + self.node_emb
        node = self.gcn(node, A_pos, A_neg)
        return self.out(node).squeeze(-1)


class PulseGCN_NoGCN(nn.Module):
    """Ablation: remove all graph convolution; no exclusivity loss is used."""

    def __init__(self, use_paaf=True):
        super().__init__()
        self.cnn = CNNFeature(use_paaf)
        self.node_fc = nn.Linear(1024, CLASS_NUM * 64)
        self.node_emb = nn.Parameter(torch.randn(CLASS_NUM, 64))
        self.out = nn.Linear(64, 1)

    def forward(self, x, A_pos=None, A_neg=None):
        node = self.node_fc(self.cnn(x)).view(-1, CLASS_NUM, 64)
        node = node + self.node_emb
        return self.out(node).squeeze(-1)


class BaseCNN(nn.Module):
    """Baseline: single-scale 1D-CNN; no graph and no exclusivity loss."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv1d(4, 128, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4))
        self.l2 = nn.Sequential(nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4))
        self.l3 = nn.Sequential(nn.Conv1d(256, 512, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.maxp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(1024, CLASS_NUM)

    def forward(self, x, A_pos=None, A_neg=None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        a = self.avg(x).squeeze(-1)
        m = self.maxp(x).squeeze(-1)
        return self.fc(torch.cat([a, m], dim=1))


class BasicLSTM(nn.Module):
    """Baseline: a basic single-layer LSTM read over the 4-channel pulse signal."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, num_layers=1,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(128, CLASS_NUM)

    def forward(self, x, A_pos=None, A_neg=None):
        x = x.permute(0, 2, 1)          # (N, 4, L) -> (N, L, 4)
        out, (h, c) = self.lstm(x)      # out: (N, L, 128)
        return self.fc(out[:, -1, :])   # take the last time step


class BasicBP(nn.Module):
    """Baseline: a basic BP network (MLP) over the flattened 4x6000 signal."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * SEQ_LEN, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, CLASS_NUM),
        )

    def forward(self, x, A_pos=None, A_neg=None):
        return self.net(x)


class FocalLoss(nn.Module):
    """Multilabel focal loss with optional exclusivity penalty."""

    def __init__(self, alpha=1.0, gamma=2.0, lam_ex=0.1, use_exclusive=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lam_ex = lam_ex
        self.use_exclusive = use_exclusive
        self.exclusive_pairs = [
            ("迟脉", "数脉"), ("浮脉", "沉脉"),
            ("虚脉", "实脉"), ("滑脉", "涩脉"),
        ]

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = (self.alpha * (1 - pt).pow(self.gamma) * bce).mean()
        if self.use_exclusive:
            ex_terms = []
            for p1, p2 in self.exclusive_pairs:
                i, j = name_to_id[p1], name_to_id[p2]
                ex_terms.append((prob[:, i] * prob[:, j]).mean())
            loss = loss + self.lam_ex * torch.stack(ex_terms).mean()
        return loss


MODEL_CONFIGS = {
    # ---- Ours (full model) and its component ablations ----
    "full": {
        "label": "Full model (Ours)",
        "use_gcn": True,
        "use_negative_graph": True,
        "use_exclusive_loss": True,
    },
    "no_gcn": {
        "label": "Ours w/o GCN",
        "use_gcn": False,
        "use_negative_graph": False,
        "use_exclusive_loss": False,
    },
    "no_neg": {
        "label": "Ours w/o exclusion graph",
        "use_gcn": True,
        "use_negative_graph": False,
        "use_exclusive_loss": False,
    },
    "relu": {
        "label": "Ours w/ ReLU (no PAAF)",
        "use_gcn": True,
        "use_negative_graph": True,
        "use_exclusive_loss": True,
    },
    "focal": {
        "label": "Ours w/ plain Focal loss",
        "use_gcn": True,
        "use_negative_graph": True,
        "use_exclusive_loss": False,
    },
    # ---- External baselines ----
    "base_cnn": {
        "label": "1D-CNN (baseline)",
        "use_gcn": False,
        "use_negative_graph": False,
        "use_exclusive_loss": False,
    },
    "lstm": {
        "label": "LSTM (baseline)",
        "use_gcn": False,
        "use_negative_graph": False,
        "use_exclusive_loss": False,
    },
    "bp": {
        "label": "BP / MLP (baseline)",
        "use_gcn": False,
        "use_negative_graph": False,
        "use_exclusive_loss": False,
    },
}


def build_model(name):
    if name == "full":
        model = PulseGCN()
    elif name == "no_gcn":
        model = PulseGCN_NoGCN()
    elif name == "no_neg":
        # Same architecture as full model; the negative graph is zeroed in training/evaluation.
        model = PulseGCN()
    elif name == "relu":
        # Same full model, but PAAF is replaced by ReLU in the CNN backbone.
        model = PulseGCN(use_paaf=False)
    elif name == "focal":
        # Same full model; the loss is reduced to a plain Focal loss in training/evaluation.
        model = PulseGCN()
    elif name == "base_cnn":
        model = BaseCNN()
    elif name == "lstm":
        model = BasicLSTM()
    elif name == "bp":
        model = BasicBP()
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(DEVICE)


def get_graphs(model_name):
    if MODEL_CONFIGS[model_name]["use_negative_graph"]:
        return A_POS_NORM, A_NEG_NORM
    return A_POS_NORM, A_NEG_ZERO


# ===================== Metrics =====================
def safe_auc(y_true, y_prob):
    values = []
    for j in range(y_true.shape[1]):
        if np.unique(y_true[:, j]).size < 2:
            continue
        try:
            values.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
        except ValueError:
            pass
    return float(np.mean(values)) if values else float("nan")


def safe_map(y_true, y_prob):
    values = []
    for j in range(y_true.shape[1]):
        if np.sum(y_true[:, j] == 1) == 0:
            continue
        try:
            values.append(average_precision_score(y_true[:, j], y_prob[:, j]))
        except ValueError:
            pass
    return float(np.mean(values)) if values else float("nan")


def calculate_metrics(y_true, y_pred, y_prob):
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro-F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Macro-Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro-Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mAP": safe_map(y_true, y_prob),
        "Macro-AUC": safe_auc(y_true, y_prob),
        "Subset-Acc": float(accuracy_score(y_true, y_pred)),
        "Hamming-Loss": float(hamming_loss(y_true, y_pred)),
    }, per_class_f1


def make_loader(X, y, indices, shuffle):
    x_tensor = torch.from_numpy(X[indices]).float()
    y_tensor = torch.from_numpy(y[indices]).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset, batch_size=BATCH, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )


def forward_model(model, x, A_pos, A_neg):
    return model(x, A_pos, A_neg)


def run_epoch_train(model, loader, criterion, optimizer, A_pos, A_neg):
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, x, A_pos, A_neg)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def run_epoch_eval(model, loader, criterion, A_pos, A_neg):
    model.eval()
    losses, probs, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = forward_model(model, x, A_pos, A_neg)
            loss = criterion(logits, y)
            p = torch.sigmoid(logits)
            losses.append(float(loss.detach().cpu()))
            probs.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(targets).astype(int)
    y_pred = (y_prob >= THRESHOLD).astype(int)
    metric_dict, per_class_f1 = calculate_metrics(y_true, y_pred, y_prob)
    return (
        float(np.mean(losses)) if losses else float("nan"),
        metric_dict["Macro-F1"], y_pred, y_prob, y_true, metric_dict, per_class_f1
    )


# ===================== Train one fold with early stopping =====================
def train_one_fold(model_name, X, y, train_idx, val_idx, repeat_idx, fold_idx):
    model = build_model(model_name)
    A_pos, A_neg = get_graphs(model_name)
    use_exclusive = MODEL_CONFIGS[model_name]["use_exclusive_loss"]
    criterion = FocalLoss(use_exclusive=use_exclusive)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Keep the outer fold untouched for final scoring. Split only the outer-training
    # data into an inner train/validation pair for early stopping.
    inner_splitter = IterativeStratification(n_splits=5, order=1)
    dummy_inner = np.zeros((len(train_idx), 1), dtype=np.float32)
    inner_train_local, inner_val_local = next(inner_splitter.split(dummy_inner, y[train_idx]))
    inner_train_idx = train_idx[inner_train_local]
    inner_val_idx = train_idx[inner_val_local]
    train_loader = make_loader(X, y, inner_train_idx, shuffle=True)
    earlystop_loader = make_loader(X, y, inner_val_idx, shuffle=False)
    outer_test_loader = make_loader(X, y, val_idx, shuffle=False)

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_val_loss = float("nan")
    best_payload = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, A_pos, A_neg)
        val_loss, val_f1, _, _, _, _, _ = run_epoch_eval(
            model, earlystop_loader, criterion, A_pos, A_neg
        )
        history.append({
            "model": model_name, "repeat": repeat_idx + 1, "fold": fold_idx + 1,
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        improved = val_f1 > best_f1 + MIN_DELTA
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            # The outer fold is not inspected for model selection.
            best_payload = True
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"  [{model_name}] repeat {repeat_idx+1}/{REPEATS} "
            f"fold {fold_idx+1}/{K} epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_Macro-F1={val_f1:.4f}"
        )

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}; best epoch={best_epoch}, best Macro-F1={best_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_dir = os.path.join(OUT_DIR, "checkpoints", model_name)
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, f"repeat_{repeat_idx+1}_fold_{fold_idx+1}.pt")
    torch.save({
        "model_name": model_name,
        "repeat": repeat_idx + 1,
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "model_state_dict": model.state_dict(),
        "config": {
            "class_num": CLASS_NUM, "epochs": EPOCHS, "batch": BATCH,
            "lr": LR, "threshold": THRESHOLD,
        },
    }, ckpt_path)

    # Final metrics are computed once on the untouched outer validation fold.
    _, _, y_pred, y_prob, y_true, best_metrics, per_class_f1 = run_epoch_eval(
        model, outer_test_loader, criterion, A_pos, A_neg
    )
    pred_dir = os.path.join(OUT_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(pred_dir, f"{model_name}_repeat_{repeat_idx+1}_fold_{fold_idx+1}.npz"),
        y_true=y_true, y_pred=y_pred, y_prob=y_prob, val_indices=val_idx,
        class_names=np.asarray(CLASS_NAMES)
    )

    return {
        "model": model_name,
        "repeat": repeat_idx + 1,
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": ckpt_path,
        **best_metrics,
    }, history, per_class_f1, y_true, y_pred, y_prob


# ===================== CSV utilities =====================
def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ===================== Plotting: four figures =====================
def _models_with_rows(rows, key="model"):
    """Return model keys in MODEL_CONFIGS order, keeping only those present in rows."""
    present = {r[key] for r in rows}
    return [m for m in MODEL_CONFIGS if m in present]


def plot_training_curves(history_rows, out_dir):
    models = _models_with_rows(history_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name in models:
        label = MODEL_CONFIGS[model_name]["label"]
        rows = [r for r in history_rows if r["model"] == model_name]
        epochs = sorted({r["epoch"] for r in rows})
        train_means, train_stds, val_means, val_stds = [], [], [], []
        f1_means, f1_stds = [], []
        for ep in epochs:
            ep_rows = [r for r in rows if r["epoch"] == ep]
            train_vals = np.asarray([r["train_loss"] for r in ep_rows], dtype=float)
            val_vals = np.asarray([r["val_loss"] for r in ep_rows], dtype=float)
            f1_vals = np.asarray([r["val_macro_f1"] for r in ep_rows], dtype=float)
            train_means.append(np.nanmean(train_vals)); train_stds.append(np.nanstd(train_vals))
            val_means.append(np.nanmean(val_vals)); val_stds.append(np.nanstd(val_vals))
            f1_means.append(np.nanmean(f1_vals)); f1_stds.append(np.nanstd(f1_vals))
        axes[0].plot(epochs, train_means, lw=1.6, label=f"{label} (train)")
        axes[0].plot(epochs, val_means, linestyle="--", lw=1.6, label=f"{label} (val)")
        axes[1].plot(epochs, f1_means, lw=1.6, label=label)
        axes[1].fill_between(
            epochs, np.asarray(f1_means) - np.asarray(f1_stds),
            np.asarray(f1_means) + np.asarray(f1_stds), alpha=0.12
        )
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25); axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_title("Validation Macro-F1 by epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].grid(True, alpha=0.25); axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(per_class_rows, out_dir):
    models = _models_with_rows(per_class_rows)
    matrix = np.full((len(models), CLASS_NUM), np.nan)
    for i, model_name in enumerate(models):
        for j, class_name in enumerate(CLASS_NAMES):
            vals = [r["f1"] for r in per_class_rows
                    if r["model"] == model_name and r["class_name"] == class_name]
            if vals:
                matrix[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(CLASS_NUM)
    width = 0.8 / max(1, len(models))
    for i, model_name in enumerate(models):
        ax.bar(x - 0.4 + width / 2 + i * width, matrix[i], width,
               label=MODEL_CONFIGS[model_name]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score across validation folds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_per_class_f1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cv_distribution(score_rows, out_dir):
    models = _models_with_rows(score_rows)
    data = [[r["Macro-F1"] for r in score_rows if r["model"] == m] for m in models]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot(data, showmeans=True)
    ax.set_xticklabels([MODEL_CONFIGS[m]["label"] for m in models], rotation=18, ha="right")
    # Overlay individual fold scores so a small number of folds remains visible.
    for i, vals in enumerate(data, start=1):
        if vals:
            x = np.full(len(vals), i, dtype=float)
            ax.scatter(x, vals, alpha=0.8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-validation Macro-F1 distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_cv_macro_f1_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(score_rows, out_dir):
    metrics_to_plot = ["Macro-F1", "Micro-F1", "mAP", "Macro-AUC"]
    models = _models_with_rows(score_rows)
    means, stds = [], []
    for metric in metrics_to_plot:
        means.append([
            np.nanmean([r[metric] for r in score_rows if r["model"] == m])
            if any(np.isfinite(r[metric]) for r in score_rows if r["model"] == m)
            else np.nan
            for m in models
        ])
        stds.append([
            np.nanstd([r[metric] for r in score_rows if r["model"] == m])
            if any(np.isfinite(r[metric]) for r in score_rows if r["model"] == m)
            else 0.0
            for m in models
        ])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)
    for i, model_name in enumerate(models):
        vals = [means[j][i] for j in range(len(metrics_to_plot))]
        err = [stds[j][i] for j in range(len(metrics_to_plot))]
        ax.bar(x - 0.4 + width / 2 + i * width, vals, width, yerr=err, capsize=2,
               label=MODEL_CONFIGS[model_name]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model and ablation comparison (mean ± standard deviation)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_model_ablation_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===================== Main CV runner =====================
def make_splits(y):
    """Create splits once and reuse them across all models for fair paired comparison."""
    splits = []
    for repeat_idx in range(REPEATS):
        rng = np.random.default_rng(SEED + repeat_idx)
        perm = rng.permutation(len(y))
        y_perm = y[perm]
        strat = IterativeStratification(n_splits=K, order=1)
        repeat_splits = []
        dummy_x = np.zeros((len(y_perm), 1), dtype=np.float32)
        for train_perm, val_perm in strat.split(dummy_x, y_perm):
            repeat_splits.append((perm[train_perm], perm[val_perm]))
        splits.append(repeat_splits)
    return splits


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUT_DIR}")
    X, y, filenames = load_all(DATA_DIR)
    print(f"Samples: {len(X)}; X shape: {X.shape}; y shape: {y.shape}")
    print("Positive samples per class:")
    for class_name, idx in PULSE_DICT.items():
        print(f"  {PULSE_EN[class_name]:<16}: {int(y[:, idx].sum())}")
    if len(X) < K:
        raise ValueError(f"Need at least K={K} samples, got {len(X)}")

    # Save the exact sample list used by this run.
    with open(os.path.join(OUT_DIR, "sample_manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "filename"])
        writer.writerows(enumerate(filenames.tolist()))

    splits = make_splits(y)
    all_score_rows, all_history_rows, per_class_rows = [], [], []

    for model_name in MODEL_CONFIGS:
        print(f"\n========== Starting {model_name} ==========")
        for repeat_idx, repeat_splits in enumerate(splits):
            for fold_idx, (train_idx, val_idx) in enumerate(repeat_splits):
                # Different but reproducible initialization per model/repeat/fold.
                set_seed(SEED + 1000 * repeat_idx + 100 * fold_idx + list(MODEL_CONFIGS).index(model_name))
                score_row, history, class_f1, y_true, y_pred, y_prob = train_one_fold(
                    model_name, X, y, train_idx, val_idx, repeat_idx, fold_idx
                )
                all_score_rows.append(score_row)
                all_history_rows.extend(history)

                for class_idx, class_name in enumerate(CLASS_NAMES):
                    per_class_rows.append({
                        "model": model_name,
                        "repeat": repeat_idx + 1,
                        "fold": fold_idx + 1,
                        "class_name": class_name,
                        "f1": float(class_f1[class_idx]),
                        "support": int(y_true[:, class_idx].sum()),
                    })

                # Incremental writes protect results if a later fold is interrupted.
                write_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), all_score_rows)
                write_csv(os.path.join(OUT_DIR, "training_history.csv"), all_history_rows)
                write_csv(os.path.join(OUT_DIR, "per_class_f1.csv"), per_class_rows)

                print(
                    f"  [{model_name}] repeat {repeat_idx+1}/{REPEATS} "
                    f"fold {fold_idx+1}/{K}: best epoch={score_row['best_epoch']}, "
                    f"Macro-F1={score_row['Macro-F1']:.4f}"
                )

    # Summary table: mean and population SD over folds.
    summary_rows = []
    metric_keys = ["Macro-F1", "Micro-F1", "Macro-Precision", "Macro-Recall",
                   "mAP", "Macro-AUC", "Subset-Acc", "Hamming-Loss", "best_epoch"]
    for model_name in MODEL_CONFIGS:
        model_rows = [r for r in all_score_rows if r["model"] == model_name]
        row = {"model": model_name, "label": MODEL_CONFIGS[model_name]["label"], "n_folds": len(model_rows)}
        for metric in metric_keys:
            vals = np.asarray([r[metric] for r in model_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_std"] = float(np.std(vals)) if len(vals) else float("nan")
        summary_rows.append(row)
    write_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), summary_rows)

    plot_training_curves(all_history_rows, OUT_DIR)
    plot_per_class_f1(per_class_rows, OUT_DIR)
    plot_cv_distribution(all_score_rows, OUT_DIR)
    plot_model_comparison(all_score_rows, OUT_DIR)

    print("\n===== Summary (mean ± std across folds) =====")
    for row in summary_rows:
        print(
            f"{row['model']:<10} "
            f"Macro-F1={row['Macro-F1_mean']:.4f} ± {row['Macro-F1_std']:.4f}; "
            f"mAP={row['mAP_mean']:.4f} ± {row['mAP_std']:.4f}; "
            f"Macro-AUC={row['Macro-AUC_mean']:.4f} ± {row['Macro-AUC_std']:.4f}"
        )
    print(f"\nFinished. Results and four figures saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()