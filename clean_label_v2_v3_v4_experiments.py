from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEATURE_DIR = DATA_DIR / "features"
OUTPUT_DIR = ROOT / "outputs" / "clean_label_v2_v3_v4"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for path in [REPORT_DIR, PRED_DIR, PLOT_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_VALUES_TORCH = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)
LABEL_VALUES_NP = np.arange(1, 6, dtype=np.float32)

CONFIG = {
    "bsd10k_metadata": METADATA_DIR / "BSD10k_metadata.csv",
    "bsd35k_metadata": METADATA_DIR / "BSD35k-CS_metadata.csv",
    "bsd10k_audio_dir": FEATURE_DIR / "clap_audio_embeddings",
    "bsd10k_text_dir": FEATURE_DIR / "clap_text_embeddings",
    "bsd35k_audio_dir": FEATURE_DIR / "BSD35k_clap_audio_embeddings",
    "bsd35k_text_dir": FEATURE_DIR / "BSD35k-CS_clap_text_embeddings",
}

RUN_CONFIG = {
    "seed": 42,
    "folds": 5,
    "epochs": 50,
    "patience": 7,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "hidden": [512, 256],
    "expected_mse_weight": 0.25,
    "history_aggregation": "pad",
    "train_final_bsd35k": True,
}

V3_THRESHOLDS_TO_EVAL = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
V3_DENSE_THRESHOLDS = np.round(np.linspace(0.05, 0.95, 181), 4)
V4_DENSE_THRESHOLDS_01 = np.round(np.linspace(0.01, 0.99, 197), 4)
V4_DENSE_THRESHOLDS_SCORE = np.round(np.linspace(1.0, 5.0, 401), 4)

V2_EXPERIMENTS = [
    {"name": "clean_v2_ce", "loss": "ce"},
    {"name": "clean_v2_ordinal_smoothing", "loss": "ordinal_smoothing"},
    {"name": "clean_v2_emd", "loss": "emd"},
    {"name": "clean_v2_expected_mse_aux", "loss": "expected_mse_aux"},
]


def finite_float32(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_metadata(df, require_confidence):
    df = df.copy()
    df["sound_id"] = df["sound_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    if "class_idx" in df.columns:
        class_idx = df["class_idx"].astype(str).str.strip()
        keep = ~((class_idx.str.len() == 3) & (class_idx.str.endswith("99") | class_idx.str.endswith("00")))
        df = df[keep].copy()
    if require_confidence:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[df["confidence"].isin([1, 2, 3, 4, 5])].copy()
        df["confidence"] = df["confidence"].astype(int)
    return df.reset_index(drop=True)


def make_class_categories(df):
    categories = sorted(df["class"].astype(str).unique().tolist())
    if len(categories) > 23:
        raise ValueError(f"Expected at most 23 classes, got {len(categories)}")
    return categories


def one_hot_23(values, categories):
    index = {category: idx for idx, category in enumerate(categories)}
    arr = np.zeros((len(values), 23), dtype=np.float32)
    for row_idx, value in enumerate(values):
        col_idx = index.get(str(value))
        if col_idx is not None:
            arr[row_idx, col_idx] = 1.0
    return arr


def load_embeddings(df, audio_dir, text_dir):
    audio_rows = []
    text_rows = []
    kept = []
    for idx, row in df.reset_index(drop=True).iterrows():
        sound_id = str(row["sound_id"])
        audio_path = audio_dir / f"{sound_id}.npy"
        text_path = text_dir / f"{sound_id}.npy"
        if audio_path.is_file() and text_path.is_file():
            audio_rows.append(finite_float32(np.load(audio_path).reshape(-1)))
            text_rows.append(finite_float32(np.load(text_path).reshape(-1)))
            kept.append(idx)
    kept_df = df.reset_index(drop=True).iloc[kept].reset_index(drop=True).copy()
    return kept_df, finite_float32(np.vstack(audio_rows)), finite_float32(np.vstack(text_rows))


def build_clean_parts(df, audio_dir, text_dir, class_categories):
    kept_df, audio, text = load_embeddings(df, audio_dir, text_dir)
    class_onehot = one_hot_23(kept_df["class"].astype(str).tolist(), class_categories)
    x = finite_float32(np.concatenate([audio, text, class_onehot], axis=1))
    return {"df": kept_df, "audio": audio, "text": text, "class_onehot": class_onehot, "x": x}


def load_bsd10k_parts():
    df = clean_metadata(pd.read_csv(CONFIG["bsd10k_metadata"]), require_confidence=True)
    class_categories = make_class_categories(df)
    parts = build_clean_parts(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"], class_categories)
    return parts, class_categories


def load_bsd35k_parts(class_categories):
    df = clean_metadata(pd.read_csv(CONFIG["bsd35k_metadata"]), require_confidence=False)
    return build_clean_parts(df, CONFIG["bsd35k_audio_dir"], CONFIG["bsd35k_text_dir"], class_categories)


class ClassificationDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(finite_float32(x), dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.x[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], output_dim),
        )

    def forward(self, x):
        return self.net(x)


def expected_score_from_probs(probs):
    probs = finite_float32(probs)
    return finite_float32((probs * LABEL_VALUES_NP.reshape(1, -1)).sum(axis=1))


def ordinal_soft_targets(y, n_classes=5, neighbor=0.05):
    target = torch.full((y.shape[0], n_classes), neighbor / (n_classes - 1), device=y.device)
    target.scatter_(1, y.view(-1, 1), 1.0 - neighbor)
    return target


def compute_v2_loss(logits, y, loss_name):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "ordinal_smoothing":
        target = ordinal_soft_targets(y, logits.shape[1])
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    probs = F.softmax(logits, dim=1)
    y_values = y.float() + 1.0
    expected = (probs * LABEL_VALUES_TORCH.view(1, -1)).sum(dim=1)
    if loss_name == "emd":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        return torch.mean(torch.sum((torch.cumsum(probs, dim=1) - torch.cumsum(target, dim=1)) ** 2, dim=1))
    if loss_name == "expected_mse_aux":
        return F.cross_entropy(logits, y) + RUN_CONFIG["expected_mse_weight"] * F.mse_loss(expected, y_values)
    raise ValueError(f"Unknown loss: {loss_name}")


def predict_v2_probs(model, loader):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(DEVICE))
            rows.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    return finite_float32(np.vstack(rows))


def evaluate_v2_loss(model, loader, loss_name):
    model.eval()
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if "y" not in batch:
                continue
            x = batch["x"].to(DEVICE)
            y = batch["y"].long().to(DEVICE)
            loss = compute_v2_loss(model(x), y, loss_name)
            total_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
    return float(total_loss / max(seen, 1))


def predict_binary_prob(model, loader):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(DEVICE)).view(-1)
            rows.append(torch.sigmoid(logits).detach().cpu().numpy())
    return finite_float32(np.concatenate(rows))


def evaluate_binary_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if "y" not in batch:
                continue
            x = batch["x"].to(DEVICE)
            y = batch["y"].float().to(DEVICE)
            loss = criterion(model(x).view(-1), y)
            total_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
    return float(total_loss / max(seen, 1))


def v2_metrics(y_true_1based, probs):
    probs = finite_float32(probs)
    score = expected_score_from_probs(probs)
    pred = probs.argmax(axis=1) + 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_1based,
        pred,
        average="macro",
        zero_division=0,
    )
    rho = spearmanr(y_true_1based, score).statistic
    return {
        "mae": float(mean_absolute_error(y_true_1based, score)),
        "hard_mae": float(mean_absolute_error(y_true_1based, pred)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "accuracy": float(accuracy_score(y_true_1based, pred)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true_1based, pred, weights="quadratic")),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def safe_auc_pr(y_true, score):
    try:
        return float(average_precision_score(y_true, score))
    except ValueError:
        return float("nan")


def safe_auc_roc(y_true, score):
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def binary_metrics(y_true, score, threshold):
    pred = (score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc_pr": safe_auc_pr(y_true, score),
        "auc_roc": safe_auc_roc(y_true, score),
    }


def train_v2_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based):
    seed_everything(RUN_CONFIG["seed"] + fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)
    train_loader = DataLoader(
        ClassificationDataset(x_train, y_0based[train_idx].astype(np.int64)),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        ClassificationDataset(x_valid, y_0based[valid_idx].astype(np.int64)),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=False,
    )

    model = MLP(
        x_raw.shape[1],
        5,
        hidden=tuple(RUN_CONFIG["hidden"]),
        dropout=RUN_CONFIG["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUN_CONFIG["epochs"])
    best_mae = math.inf
    best_state = None
    best_probs = None
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(RUN_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].long().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_v2_loss(model(x), y, exp["loss"])
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        probs = predict_v2_probs(model, valid_loader)
        val_loss = evaluate_v2_loss(model, valid_loader, exp["loss"])
        pred = probs.argmax(axis=1) + 1
        val_mae = float(mean_absolute_error(y_1based[valid_idx], expected_score_from_probs(probs)))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(seen, 1),
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_mae < best_mae - 1e-6:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_probs = probs
            stale = 0
        else:
            stale += 1
            if stale >= RUN_CONFIG["patience"]:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = max(len(history) - 1, 0)
        best_probs = predict_v2_probs(model, valid_loader)
    model.load_state_dict(best_state)
    ckpt_dir = CHECKPOINT_DIR / exp["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "input_dim": int(x_raw.shape[1]),
            "experiment": exp,
            "best_epoch": int(best_epoch),
            "run_config": RUN_CONFIG,
        },
        ckpt_dir / f"fold_{fold}.pt",
    )
    return best_probs, history, {"fold": fold, "best_epoch": int(best_epoch), "best_mae": float(best_mae)}


def train_binary_fold(fold, train_idx, valid_idx, x_raw, y_binary):
    seed_everything(RUN_CONFIG["seed"] + fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)
    y_train = y_binary[train_idx].astype(np.float32)
    y_valid = y_binary[valid_idx].astype(np.float32)
    train_loader = DataLoader(
        ClassificationDataset(x_train, y_train),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        ClassificationDataset(x_valid, y_valid),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=False,
    )
    model = MLP(
        x_raw.shape[1],
        1,
        hidden=tuple(RUN_CONFIG["hidden"]),
        dropout=RUN_CONFIG["dropout"],
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUN_CONFIG["epochs"])
    best_f1 = -1.0
    best_state = None
    best_prob = None
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(RUN_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].float().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).view(-1), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        prob = predict_binary_prob(model, valid_loader)
        val_loss = evaluate_binary_loss(model, valid_loader, criterion)
        best_epoch_threshold = pick_f1_optimal(y_binary[valid_idx], prob, V3_DENSE_THRESHOLDS)[0]
        val_f1 = float(best_epoch_threshold["f1"])
        val_auc_pr = safe_auc_pr(y_binary[valid_idx], prob)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(seen, 1),
                "val_loss": val_loss,
                "val_f1_best_threshold": val_f1,
                "val_auc_pr": val_auc_pr,
                "best_epoch_threshold": float(best_epoch_threshold["threshold"]),
            }
        )
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_prob = prob
            stale = 0
        else:
            stale += 1
            if stale >= RUN_CONFIG["patience"]:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = max(len(history) - 1, 0)
        best_prob = predict_binary_prob(model, valid_loader)
    model.load_state_dict(best_state)
    ckpt_dir = CHECKPOINT_DIR / "clean_v3_binary"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "input_dim": int(x_raw.shape[1]),
            "best_epoch": int(best_epoch),
            "run_config": RUN_CONFIG,
        },
        ckpt_dir / f"fold_{fold}.pt",
    )
    best_fold_threshold = pick_f1_optimal(y_binary[valid_idx], best_prob, V3_DENSE_THRESHOLDS)[0]
    return best_prob, history, {
        "fold": fold,
        "best_epoch": int(best_epoch),
        "best_f1": float(best_f1),
        "best_epoch_threshold": float(best_fold_threshold["threshold"]),
    }


def aggregate_histories(histories):
    lengths = [len(history) for history in histories]
    if not lengths:
        return pd.DataFrame()
    target_len = min(lengths) if RUN_CONFIG["history_aggregation"] == "truncate" else max(lengths)
    rows = []
    for idx in range(target_len):
        row = {"epoch": idx + 1}
        for key in ["train_loss", "val_loss", "val_mae", "val_f1_best_threshold", "val_auc_pr"]:
            vals = []
            for history in histories:
                source_idx = min(idx, len(history) - 1)
                if key in history[source_idx]:
                    vals.append(history[source_idx][key])
            if vals:
                row[f"{key}_mean"] = float(np.mean(vals))
                row[f"{key}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def save_history_outputs(name, histories):
    for fold, history in enumerate(histories):
        pd.DataFrame(history).to_csv(REPORT_DIR / f"{name}_fold_{fold}_history.csv", index=False)
    aggregate = aggregate_histories(histories)
    aggregate.to_csv(REPORT_DIR / f"{name}_mean_history.csv", index=False)
    if plt is None or aggregate.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(aggregate["epoch"], aggregate["train_loss_mean"], color="#1f77b4", label="Train loss")
    ax1.fill_between(
        aggregate["epoch"],
        aggregate["train_loss_mean"] - aggregate.get("train_loss_std", 0.0),
        aggregate["train_loss_mean"] + aggregate.get("train_loss_std", 0.0),
        color="#1f77b4",
        alpha=0.15,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(aggregate["epoch"], aggregate["val_loss_mean"], color="#d62728", label="Val loss")
    ax2.fill_between(
        aggregate["epoch"],
        aggregate["val_loss_mean"] - aggregate.get("val_loss_std", 0.0),
        aggregate["val_loss_mean"] + aggregate.get("val_loss_std", 0.0),
        color="#d62728",
        alpha=0.12,
    )
    ax2.set_ylabel("Val loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    plt.title(f"{name} - Fold-mean train and validation loss")
    fig.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_train_loss_val_loss.png", dpi=160)
    plt.close(fig)


def save_confusion(name, y_true, y_pred, labels, display_labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    display_labels = display_labels or labels
    pd.DataFrame(cm, index=[f"true_{x}" for x in display_labels], columns=[f"pred_{x}" for x in display_labels]).to_csv(
        REPORT_DIR / f"{name}_confusion_counts.csv"
    )
    pd.DataFrame(
        cm_norm,
        index=[f"true_{x}" for x in display_labels],
        columns=[f"pred_{x}" for x in display_labels],
    ).to_csv(REPORT_DIR / f"{name}_confusion_row_normalized.csv")
    if plt is None:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.title(f"{name} - Row-normalized confusion")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(display_labels)), display_labels)
    plt.yticks(range(len(display_labels)), display_labels)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{name}_confusion_row_normalized.png", dpi=160)
    plt.close()


def run_v2(parts, y_0based, y_1based, splits):
    x_raw = parts["x"]
    all_fold_rows = []
    oof_payloads = {}
    for exp in V2_EXPERIMENTS:
        histories = []
        oof_probs = np.zeros((len(y_1based), 5), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            probs, history, fold_info = train_v2_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based)
            oof_probs[valid_idx] = probs
            histories.append(history)
            row = {"experiment": exp["name"], "feature_dim": int(x_raw.shape[1]), **fold_info}
            row.update(v2_metrics(y_1based[valid_idx], probs))
            all_fold_rows.append(row)
            print(f"{exp['name']} fold={fold} mae={row['mae']:.4f} acc={row['accuracy']:.4f}")
        save_history_outputs(exp["name"], histories)
        pred = oof_probs.argmax(axis=1) + 1
        save_confusion(exp["name"], y_1based, pred, labels=[1, 2, 3, 4, 5])
        oof_payloads[exp["name"]] = {
            "probs": oof_probs,
            "score": expected_score_from_probs(oof_probs),
            "pred_class": pred,
        }

    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(REPORT_DIR / "clean_v2_fold_metrics.csv", index=False)
    rows = []
    for exp in V2_EXPERIMENTS:
        sub = fold_df[fold_df["experiment"] == exp["name"]]
        row = {"experiment": exp["name"], "feature_dim": int(sub["feature_dim"].iloc[0])}
        for metric in ["mae", "hard_mae", "accuracy", "quadratic_weighted_kappa", "macro_precision", "macro_recall", "macro_f1"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["mae_mean", "macro_f1_mean"], ascending=[True, False])
    summary.to_csv(REPORT_DIR / "clean_v2_summary.csv", index=False)
    best_name = str(summary.iloc[0]["experiment"])
    best = oof_payloads[best_name]
    out = parts["df"][["sound_id", "class", "confidence"]].copy()
    for i in range(5):
        out[f"prob_confidence_{i + 1}"] = best["probs"][:, i]
    out["predicted_confidence_score"] = best["score"]
    out["predicted_confidence_class"] = best["pred_class"]
    out["source_experiment"] = best_name
    out.to_csv(PRED_DIR / "BSD10k_oof_clean_v2.csv", index=False)
    return summary, fold_df, oof_payloads, best_name


def run_v3(parts, y_binary, splits):
    x_raw = parts["x"]
    histories = []
    oof_prob = np.zeros(len(y_binary), dtype=np.float32)
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        prob, history, fold_info = train_binary_fold(fold, train_idx, valid_idx, x_raw, y_binary)
        oof_prob[valid_idx] = prob
        histories.append(history)
        threshold_metrics = binary_metrics(y_binary[valid_idx], prob, float(fold_info["best_epoch_threshold"]))
        row = {"experiment": "clean_v3_binary", "feature_dim": int(x_raw.shape[1]), **fold_info, **threshold_metrics}
        fold_rows.append(row)
        print(f"clean_v3_binary fold={fold} f1={row['f1']:.4f} auc_pr={row['auc_pr']:.4f}")
    save_history_outputs("clean_v3_binary", histories)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(REPORT_DIR / "clean_v3_binary_fold_metrics.csv", index=False)
    fixed_threshold_sweep = threshold_sweep(y_binary, oof_prob, V3_THRESHOLDS_TO_EVAL)
    dense_threshold_sweep = threshold_sweep(y_binary, oof_prob, V3_DENSE_THRESHOLDS)
    f1_optimal = dense_threshold_sweep.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
    precision_candidates = dense_threshold_sweep[dense_threshold_sweep["recall"] >= 0.7].copy()
    precision_optimal_recall70 = precision_candidates.sort_values(
        ["precision", "f1", "threshold"], ascending=[False, False, False]
    ).iloc[0]
    summary_rows = [
        {"model": "Binary MLP @ F1-optimal", **binary_metrics(y_binary, oof_prob, float(f1_optimal["threshold"]))},
        {
            "model": "Binary MLP @ precision-optimal recall>=0.7",
            **binary_metrics(y_binary, oof_prob, float(precision_optimal_recall70["threshold"])),
        },
        {"model": "Binary MLP @ threshold 0.5 reference", **binary_metrics(y_binary, oof_prob, 0.5)},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.insert(1, "feature_dim", int(x_raw.shape[1]))
    fixed_threshold_sweep.to_csv(REPORT_DIR / "clean_v3_fixed_threshold_sweep.csv", index=False)
    dense_threshold_sweep.to_csv(REPORT_DIR / "clean_v3_dense_threshold_sweep.csv", index=False)
    summary.to_csv(REPORT_DIR / "clean_v3_binary_summary.csv", index=False)
    save_confusion(
        "clean_v3_binary_f1_optimal",
        y_binary,
        (oof_prob >= float(f1_optimal["threshold"])).astype(int),
        labels=[0, 1],
        display_labels=["low", "high"],
    )
    out = parts["df"][["sound_id", "class", "confidence"]].copy()
    out["target_high_confidence"] = y_binary.astype(int)
    out["predicted_high_confidence_prob"] = oof_prob
    out["predicted_confidence_score"] = 1.0 + 4.0 * oof_prob
    out["binary_f1_optimal_threshold"] = float(f1_optimal["threshold"])
    out["binary_precision_optimal_recall70_threshold"] = float(precision_optimal_recall70["threshold"])
    out.to_csv(PRED_DIR / "BSD10k_oof_clean_v3_binary.csv", index=False)
    return summary, fold_df, oof_prob


def percentile_rank(values):
    s = pd.Series(values)
    return s.rank(method="average", pct=True).to_numpy(dtype=np.float32)


def threshold_sweep(y_true, score, thresholds):
    return pd.DataFrame([binary_metrics(y_true, score, threshold) for threshold in thresholds])


def pick_f1_optimal(y_true, score, thresholds):
    sweep = threshold_sweep(y_true, score, thresholds)
    best = sweep.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
    return best, sweep


def pick_precision_optimal_at_recall(y_true, score, thresholds, min_recall=0.7):
    sweep = threshold_sweep(y_true, score, thresholds)
    candidates = sweep[sweep["recall"] >= min_recall].copy()
    if candidates.empty:
        candidates = sweep.copy()
    best = candidates.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    return best, sweep


def run_v4(parts, y_binary, v2_payloads, best_v2_name, v3_prob, splits):
    probs = v2_payloads[best_v2_name]["probs"]
    score = v2_payloads[best_v2_name]["score"]
    df = parts["df"][["sound_id", "class", "confidence"]].copy()
    df["target_high_confidence"] = y_binary.astype(int)
    df["binary_mlp_prob"] = v3_prob
    for idx in range(5):
        df[f"prob_confidence_{idx + 1}"] = probs[:, idx]
    df["fiveclass_score"] = score
    df["fiveclass_score_01"] = ((score - 1.0) / 4.0).clip(0, 1)
    df["fiveclass_p45"] = probs[:, 3] + probs[:, 4]
    df["fiveclass_margin_p45_minus_p3"] = df["fiveclass_p45"] - probs[:, 2]
    df["rank_average_binary_p45"] = (percentile_rank(df["binary_mlp_prob"]) + percentile_rank(df["fiveclass_p45"])) / 2.0
    df["rank_average_binary_score"] = (percentile_rank(df["binary_mlp_prob"]) + percentile_rank(df["fiveclass_score"])) / 2.0

    score_specs = [
        ("binary_mlp_prob", "Binary MLP probability", V4_DENSE_THRESHOLDS_01),
        ("fiveclass_p45", "5-class P4+P5", V4_DENSE_THRESHOLDS_01),
        ("fiveclass_score", "5-class expected score", V4_DENSE_THRESHOLDS_SCORE),
        ("fiveclass_score_01", "5-class expected score scaled 0-1", V4_DENSE_THRESHOLDS_01),
        ("fiveclass_margin_p45_minus_p3", "5-class margin P45-P3", np.round(np.linspace(-1.0, 1.0, 401), 4)),
        ("rank_average_binary_p45", "Rank average: binary + P45", V4_DENSE_THRESHOLDS_01),
        ("rank_average_binary_score", "Rank average: binary + expected score", V4_DENSE_THRESHOLDS_01),
    ]

    summary_rows = []
    sweeps = []
    for col, label, thresholds in score_specs:
        score_values = df[col].to_numpy(dtype=float)
        best, sweep = pick_f1_optimal(y_binary, score_values, thresholds)
        prec_best, _ = pick_precision_optimal_at_recall(y_binary, score_values, thresholds, min_recall=0.7)
        summary_rows.append({"method": label, "score_column": col, "selection": "F1-optimal", **best.to_dict()})
        summary_rows.append(
            {
                "method": label,
                "score_column": col,
                "selection": "precision-optimal @ recall>=0.7",
                **prec_best.to_dict(),
            }
        )
        sweep = sweep.copy()
        sweep["method"] = label
        sweep["score_column"] = col
        sweeps.append(sweep)

    stack_features = [
        "binary_mlp_prob",
        "fiveclass_score_01",
        "fiveclass_p45",
        "prob_confidence_3",
        "prob_confidence_4",
        "prob_confidence_5",
        "fiveclass_margin_p45_minus_p3",
    ]
    x_stack = df[stack_features].to_numpy(dtype=np.float32)
    stack_oof = np.zeros(len(df), dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(splits):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs"),
        )
        model.fit(x_stack[train_idx], y_binary[train_idx])
        stack_oof[valid_idx] = model.predict_proba(x_stack[valid_idx])[:, 1]
    df["logistic_stacker_prob"] = stack_oof
    best_stack, stack_sweep = pick_f1_optimal(y_binary, stack_oof, V4_DENSE_THRESHOLDS_01)
    prec_best_stack, _ = pick_precision_optimal_at_recall(y_binary, stack_oof, V4_DENSE_THRESHOLDS_01, min_recall=0.7)
    summary_rows.append({"method": "OOF logistic stacker", "score_column": "logistic_stacker_prob", "selection": "F1-optimal", **best_stack.to_dict()})
    summary_rows.append(
        {
            "method": "OOF logistic stacker",
            "score_column": "logistic_stacker_prob",
            "selection": "precision-optimal @ recall>=0.7",
            **prec_best_stack.to_dict(),
        }
    )
    stack_sweep = stack_sweep.copy()
    stack_sweep["method"] = "OOF logistic stacker"
    stack_sweep["score_column"] = "logistic_stacker_prob"
    sweeps.append(stack_sweep)

    summary = pd.DataFrame(summary_rows).sort_values(["selection", "f1", "precision"], ascending=[True, False, False])
    sweep_df = pd.concat(sweeps, ignore_index=True)
    summary.to_csv(REPORT_DIR / "clean_v4_score_stacking_summary.csv", index=False)
    sweep_df.to_csv(REPORT_DIR / "clean_v4_threshold_sweeps.csv", index=False)
    df.to_csv(PRED_DIR / "BSD10k_oof_clean_v4_scores.csv", index=False)

    f1_candidates = summary[summary["selection"] == "F1-optimal"].copy()
    best_row = f1_candidates.sort_values(["f1", "auc_pr"], ascending=[False, False]).iloc[0]
    best_col = str(best_row["score_column"])
    best_threshold = float(best_row["threshold"])
    best_pred = (df[best_col].to_numpy(dtype=float) >= best_threshold).astype(int)
    save_confusion(
        "clean_v4_best_binary",
        y_binary,
        best_pred,
        labels=[0, 1],
        display_labels=["low", "high"],
    )
    precision_candidates = summary[summary["selection"] == "precision-optimal @ recall>=0.7"].copy()
    precision_row = precision_candidates.sort_values(["precision", "f1"], ascending=[False, False]).iloc[0]
    precision_pred = (
        df[str(precision_row["score_column"])].to_numpy(dtype=float) >= float(precision_row["threshold"])
    ).astype(int)
    save_confusion(
        "clean_v4_precision_recall70_binary",
        y_binary,
        precision_pred,
        labels=[0, 1],
        display_labels=["low", "high"],
    )

    if plt is not None:
        plt.figure(figsize=(8, 4.5))
        for method in summary["method"].head(5):
            sub = sweep_df[sweep_df["method"] == method]
            plt.plot(sub["threshold"], sub["f1"], label=method)
        plt.title("clean v4 - F1 by threshold")
        plt.xlabel("Threshold")
        plt.ylabel("F1")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "clean_v4_threshold_f1.png", dpi=160)
        plt.close()

    return summary, df, stack_features


def train_final_v2_model(exp, parts, y_0based, train_epochs):
    seed_everything(RUN_CONFIG["seed"] + 999)
    x_raw = parts["x"]
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)
    loader = DataLoader(
        ClassificationDataset(x, y_0based.astype(np.int64)),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=True,
    )
    model = MLP(x_raw.shape[1], 5, hidden=tuple(RUN_CONFIG["hidden"]), dropout=RUN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_epochs, 1))
    for _ in range(max(train_epochs, 1)):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_v2_loss(model(batch["x"].to(DEVICE)), batch["y"].long().to(DEVICE), exp["loss"])
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model, scaler


def predict_final_v2(model, scaler, parts):
    x = scaler.transform(parts["x"]).astype(np.float32)
    loader = DataLoader(ClassificationDataset(x), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
    return predict_v2_probs(model, loader)


def train_final_binary_model(parts, y_binary, train_epochs):
    seed_everything(RUN_CONFIG["seed"] + 1999)
    x_raw = parts["x"]
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)
    y = y_binary.astype(np.float32)
    loader = DataLoader(ClassificationDataset(x, y), batch_size=RUN_CONFIG["batch_size"], shuffle=True)
    model = MLP(x_raw.shape[1], 1, hidden=tuple(RUN_CONFIG["hidden"]), dropout=RUN_CONFIG["dropout"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_epochs, 1))
    for _ in range(max(train_epochs, 1)):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch["x"].to(DEVICE)).view(-1), batch["y"].float().to(DEVICE))
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model, scaler


def predict_final_binary(model, scaler, parts):
    x = scaler.transform(parts["x"]).astype(np.float32)
    loader = DataLoader(ClassificationDataset(x), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
    return predict_binary_prob(model, loader)


def load_binary_fold_models():
    artifacts = []
    for ckpt_path in sorted((CHECKPOINT_DIR / "clean_v3_binary").glob("fold_*.pt")):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        model = MLP(
            ckpt["input_dim"],
            1,
            hidden=tuple(ckpt["run_config"]["hidden"]),
            dropout=ckpt["run_config"]["dropout"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"].astype(np.float64)
        scaler.scale_ = ckpt["scaler_scale"].astype(np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = ckpt["input_dim"]
        artifacts.append((model, scaler, ckpt))
    if not artifacts:
        raise FileNotFoundError("No clean v3 binary fold checkpoints found. Run clean v3 CV first.")
    return artifacts


def predict_binary_fold_ensemble(parts_to_predict):
    fold_probs = []
    for model, scaler, _ckpt in load_binary_fold_models():
        x_scaled = scaler.transform(parts_to_predict["x"]).astype(np.float32)
        loader = DataLoader(ClassificationDataset(x_scaled), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
        fold_probs.append(predict_binary_prob(model, loader))
    return finite_float32(np.vstack(fold_probs).mean(axis=0))


def predict_bsd35k(class_categories, parts, y_0based, y_binary, best_v2_name, stack_features):
    if not RUN_CONFIG["train_final_bsd35k"]:
        return None
    bsd35_parts = load_bsd35k_parts(class_categories)
    best_exp = next(exp for exp in V2_EXPERIMENTS if exp["name"] == best_v2_name)
    history_files = sorted(REPORT_DIR.glob(f"{best_v2_name}_fold_*_history.csv"))
    best_epochs = []
    for path in history_files:
        hist = pd.read_csv(path)
        if "val_mae" in hist:
            best_epochs.append(int(hist.loc[hist["val_mae"].idxmin(), "epoch"]))
    train_epochs = max(1, min(RUN_CONFIG["epochs"], int(round(np.median(best_epochs))) if best_epochs else RUN_CONFIG["epochs"]))
    v2_model, v2_scaler = train_final_v2_model(best_exp, parts, y_0based, train_epochs)
    v2_probs = predict_final_v2(v2_model, v2_scaler, bsd35_parts)

    binary_prob = predict_binary_fold_ensemble(bsd35_parts)

    out = bsd35_parts["df"][["sound_id", "class"]].copy()
    for i in range(5):
        out[f"prob_confidence_{i + 1}"] = v2_probs[:, i]
    out["binary_mlp_prob"] = binary_prob
    out["fiveclass_score"] = expected_score_from_probs(v2_probs)
    out["fiveclass_score_01"] = ((out["fiveclass_score"] - 1.0) / 4.0).clip(0, 1)
    out["fiveclass_p45"] = v2_probs[:, 3] + v2_probs[:, 4]
    out["fiveclass_margin_p45_minus_p3"] = out["fiveclass_p45"] - v2_probs[:, 2]
    out["rank_average_binary_p45"] = (percentile_rank(out["binary_mlp_prob"]) + percentile_rank(out["fiveclass_p45"])) / 2.0
    out["rank_average_binary_score"] = (percentile_rank(out["binary_mlp_prob"]) + percentile_rank(out["fiveclass_score"])) / 2.0

    oof_v4 = pd.read_csv(PRED_DIR / "BSD10k_oof_clean_v4_scores.csv")
    final_stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    final_stacker.fit(oof_v4[stack_features].to_numpy(dtype=np.float32), y_binary)
    out["logistic_stacker_prob"] = final_stacker.predict_proba(out[stack_features].to_numpy(dtype=np.float32))[:, 1]

    v4_summary = pd.read_csv(REPORT_DIR / "clean_v4_score_stacking_summary.csv")
    f1_candidates = v4_summary[v4_summary["selection"] == "F1-optimal"].copy()
    best_row = f1_candidates.sort_values(["f1", "auc_pr"], ascending=[False, False]).iloc[0]
    best_col = str(best_row["score_column"])
    best_threshold = float(best_row["threshold"])
    out["v4_filter_score"] = out[best_col].astype(float)
    out["v4_best_method"] = str(best_row["method"])
    out["v4_best_threshold"] = best_threshold
    out["v4_predicted_high_confidence"] = (out["v4_filter_score"] >= out["v4_best_threshold"]).astype(int)
    out["predicted_confidence_score"] = 1.0 + 4.0 * out["v4_filter_score"].clip(0.0, 1.0)

    oof_v4 = pd.read_csv(PRED_DIR / "BSD10k_oof_clean_v4_scores.csv")
    if best_col == "fiveclass_score":
        scenario_thresholds = sorted(set([3.0, 3.25, 3.5, 3.75, 4.0, best_threshold]))
    elif best_col == "fiveclass_margin_p45_minus_p3":
        scenario_thresholds = sorted(set([-0.2, 0.0, 0.2, 0.4, 0.6, best_threshold]))
    else:
        scenario_thresholds = sorted(set([0.5, 0.6, 0.7, 0.8, 0.9, best_threshold]))
    scenario_rows = []
    for threshold in scenario_thresholds:
        oof_metric = binary_metrics(y_binary, oof_v4[best_col].to_numpy(dtype=float), float(threshold))
        scenario_rows.append(
            {
                "method": str(best_row["method"]),
                "score_column": best_col,
                "threshold": float(threshold),
                "retained_samples": int((out["v4_filter_score"] >= threshold).sum()),
                "retained_ratio": float((out["v4_filter_score"] >= threshold).mean()),
                "expected_precision_from_oof": oof_metric["precision"],
                "expected_recall_from_oof": oof_metric["recall"],
                "expected_f1_from_oof": oof_metric["f1"],
            }
        )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(REPORT_DIR / "clean_v4_bsd35k_filtering_scenarios.csv", index=False)
    scenario_df.to_csv(PRED_DIR / "BSD35k-CS_clean_v4_threshold_scenarios.csv", index=False)
    out.to_csv(PRED_DIR / "BSD35k-CS_clean_v4_scores.csv", index=False)
    return out


def write_run_manifest(result):
    manifest = {
        "device": str(DEVICE),
        "run_config": RUN_CONFIG,
        "feature_definition": "clean_label = audio(512) + text(512) + class_onehot(23); no top_class, no metadata length",
        "feature_dim": int(result["feature_dim"]),
        "best_v2_experiment": result["best_v2_name"],
        "outputs": {
            "reports": str(REPORT_DIR),
            "predictions": str(PRED_DIR),
            "plots": str(PLOT_DIR),
        },
    }
    with open(REPORT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    started = time.time()
    seed_everything(RUN_CONFIG["seed"])
    parts, class_categories = load_bsd10k_parts()
    y_1based = parts["df"]["confidence"].to_numpy(dtype=np.int64)
    y_0based = y_1based - 1
    y_binary = (y_1based >= 4).astype(np.int64)
    v2_splits = list(
        StratifiedKFold(
            n_splits=RUN_CONFIG["folds"],
            shuffle=True,
            random_state=RUN_CONFIG["seed"],
        ).split(parts["x"], y_1based)
    )
    v3_splits = list(
        StratifiedKFold(
            n_splits=RUN_CONFIG["folds"],
            shuffle=True,
            random_state=RUN_CONFIG["seed"],
        ).split(np.zeros(len(y_binary)), y_binary)
    )

    print("clean_label feature shape:", parts["x"].shape)
    print("device:", DEVICE)
    v2_summary, v2_fold_df, v2_payloads, best_v2_name = run_v2(parts, y_0based, y_1based, v2_splits)
    v3_summary, v3_fold_df, v3_prob = run_v3(parts, y_binary, v3_splits)
    v4_summary, v4_oof, stack_features = run_v4(parts, y_binary, v2_payloads, best_v2_name, v3_prob, v3_splits)
    bsd35 = predict_bsd35k(class_categories, parts, y_0based, y_binary, best_v2_name, stack_features)

    result = {
        "feature_dim": int(parts["x"].shape[1]),
        "v2_summary": v2_summary,
        "v2_fold_df": v2_fold_df,
        "v3_summary": v3_summary,
        "v3_fold_df": v3_fold_df,
        "v4_summary": v4_summary,
        "v4_oof": v4_oof,
        "best_v2_name": best_v2_name,
        "bsd35": bsd35,
        "elapsed_seconds": time.time() - started,
    }
    write_run_manifest(result)
    print("best clean v2:", best_v2_name)
    print("best clean v4:", v4_summary.iloc[0]["method"], "threshold=", v4_summary.iloc[0]["threshold"])
    print(f"elapsed: {result['elapsed_seconds']:.1f}s")
    return result


if __name__ == "__main__":
    main()
