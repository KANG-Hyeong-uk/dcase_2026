from pathlib import Path
import json
import math
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEATURE_DIR = DATA_DIR / "features"
OUTPUT_DIR = ROOT / "outputs" / "confidence_model_v5"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"

for path in [REPORT_DIR, PRED_DIR, CHECKPOINT_DIR, PLOT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_VALUES_NP = np.arange(1, 6, dtype=np.float32)
LABEL_VALUES_TORCH = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)

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
    "epochs": 60,
    "patience": 8,
    "batch_size": 256,
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "dropout": 0.30,
    "hidden": [512, 256],
    "tree_estimators": 250,
    "tree_learning_rate": 0.04,
    "history_aggregation": "pad",  # "pad" or "truncate"
    "use_class_sample_weights": True,
    "class_weight_source": "metadata",  # "metadata" is safest when supplied counts do not exactly match the CSV.
    "train_final_models": True,
    "predict_bsd35k": True,
}

SUPPLIED_DATASET_DISTRIBUTION = {
    "fx-a": 166,
    "fx-el": 208,
    "fx-ex": 297,
    "fx-h": 975,
    "fx-m": 279,
    "fx-n": 652,
    "fx-o": 1211,
    "fx-v": 201,
    "is-e": 297,
    "is-k": 362,
    "is-p": 606,
    "is-s": 528,
    "is-w": 566,
    "m-m": 323,
    "m-si": 721,
    "m-sp": 683,
    "sp-c": 177,
    "sp-p": 364,
    "sp-s": 806,
    "ss-i": 207,
    "ss-n": 391,
    "ss-s": 217,
    "ss-u": 719,
}

EXPERIMENTS = [
    {
        "name": "E1_base_mlp",
        "label": "E1 (Base MLP)",
        "feature_set": "E1",
        "model_type": "ordinal_mlp",
        "description": "Audio + text + class one-hot, no top_class/meta length features.",
    },
    {
        "name": "E2_agreement_mlp",
        "label": "E2 (+ Agreement)",
        "feature_set": "E2",
        "model_type": "ordinal_mlp",
        "description": "E1 plus cosine similarity, L2 distance, and dot product.",
    },
    {
        "name": "E3_proto_consistency_mlp",
        "label": "E3 (+ Proto/Cons)",
        "feature_set": "E3",
        "model_type": "ordinal_mlp",
        "description": "E2 plus OOF prototype and auxiliary class-consistency scalars.",
    },
    {
        "name": "E4_tree_scalar",
        "label": "E4 (Tree Scalar)",
        "feature_set": "E4",
        "model_type": "tree_regressor",
        "description": "Only 10 scalar features plus class one-hot.",
    },
]


def clean_metadata(df, require_confidence):
    df = df.copy()
    df["sound_id"] = df["sound_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    if "class_top" not in df.columns:
        df["class_top"] = df["class"].str.split("-").str[0]
    else:
        fallback = df["class"].str.split("-").str[0]
        df["class_top"] = df["class_top"].fillna(fallback).astype(str).str.strip()

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
        raise ValueError(f"Expected at most 23 class categories, got {len(categories)}")
    return categories


def one_hot_23(values, categories):
    index = {category: idx for idx, category in enumerate(categories)}
    arr = np.zeros((len(values), 23), dtype=np.float32)
    for row_idx, value in enumerate(values):
        col_idx = index.get(str(value))
        if col_idx is not None:
            arr[row_idx, col_idx] = 1.0
    return arr


def finite_float32(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


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


def normalized_rows(x):
    x = finite_float32(x)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return finite_float32(x / np.maximum(norm, 1e-8))


def build_parts(df, audio_dir, text_dir, class_categories):
    kept_df, audio, text = load_embeddings(df, audio_dir, text_dir)
    class_onehot = one_hot_23(kept_df["class"].astype(str).tolist(), class_categories)
    class_ids = class_onehot.argmax(axis=1).astype(np.int64)
    product = finite_float32(audio * text)
    dot = finite_float32(product.sum(axis=1, keepdims=True))
    audio_norm = finite_float32(np.linalg.norm(audio, axis=1, keepdims=True))
    text_norm = finite_float32(np.linalg.norm(text, axis=1, keepdims=True))
    cosine = finite_float32(dot / np.maximum(audio_norm * text_norm, 1e-8))
    l2 = finite_float32(np.linalg.norm(audio - text, axis=1, keepdims=True))
    agreement = finite_float32(np.hstack([cosine, l2, dot]))
    return {
        "df": kept_df,
        "audio": audio,
        "text": text,
        "class": class_onehot,
        "class_ids": class_ids,
        "agreement": agreement,
    }


def load_bsd10k_parts():
    df = clean_metadata(pd.read_csv(CONFIG["bsd10k_metadata"]), require_confidence=True)
    class_categories = make_class_categories(df)
    parts = build_parts(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"], class_categories)
    return parts, class_categories


def load_bsd35k_parts(class_categories):
    df = clean_metadata(pd.read_csv(CONFIG["bsd35k_metadata"]), require_confidence=False)
    return build_parts(df, CONFIG["bsd35k_audio_dir"], CONFIG["bsd35k_text_dir"], class_categories)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean_class_prototypes(x_norm, class_ids, train_idx, n_classes=23):
    prototypes = np.zeros((n_classes, x_norm.shape[1]), dtype=np.float32)
    for cls in range(n_classes):
        mask = class_ids[train_idx] == cls
        if mask.any():
            proto = x_norm[train_idx][mask].mean(axis=0)
            prototypes[cls] = proto / max(np.linalg.norm(proto), 1e-8)
    return prototypes


def prototype_features_for_indices(parts, train_idx, target_idx):
    class_ids = parts["class_ids"]
    audio_norm = normalized_rows(parts["audio"])
    text_norm = normalized_rows(parts["text"])
    audio_proto = mean_class_prototypes(audio_norm, class_ids, train_idx)
    text_proto = mean_class_prototypes(text_norm, class_ids, train_idx)

    audio_scores = audio_norm[target_idx] @ audio_proto.T
    text_scores = text_norm[target_idx] @ text_proto.T
    label_ids = class_ids[target_idx]
    row = np.arange(len(target_idx))

    audio_label = audio_scores[row, label_ids]
    text_label = text_scores[row, label_ids]
    audio_other = audio_scores.copy()
    text_other = text_scores.copy()
    audio_other[row, label_ids] = -np.inf
    text_other[row, label_ids] = -np.inf
    audio_margin = audio_label - np.max(audio_other, axis=1)
    text_margin = text_label - np.max(text_other, axis=1)
    return finite_float32(np.column_stack([audio_label, text_label, audio_margin, text_margin]))


def fit_auxiliary_class_model(parts, train_idx, seed):
    x_train = np.hstack([parts["audio"][train_idx], parts["text"][train_idx]]).astype(np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(x_train, parts["class_ids"][train_idx])
    return scaler, clf


def auxiliary_consistency_features(parts, target_idx, scaler, clf):
    x_target = np.hstack([parts["audio"][target_idx], parts["text"][target_idx]]).astype(np.float32)
    probs_small = clf.predict_proba(scaler.transform(x_target).astype(np.float32))
    probs = np.zeros((len(target_idx), 23), dtype=np.float32)
    probs[:, clf.classes_.astype(int)] = probs_small.astype(np.float32)
    label_ids = parts["class_ids"][target_idx]
    row = np.arange(len(target_idx))
    label_prob = probs[row, label_ids]
    top1_prob = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=1)
    return finite_float32(np.column_stack([label_prob, top1_prob, entropy]))


def build_oof_scalar_features(parts, splits):
    scalar = np.zeros((len(parts["df"]), 10), dtype=np.float32)
    scalar[:, :3] = parts["agreement"]
    for fold, (train_idx, valid_idx) in enumerate(splits):
        proto = prototype_features_for_indices(parts, train_idx, valid_idx)
        scaler, clf = fit_auxiliary_class_model(parts, train_idx, RUN_CONFIG["seed"] + fold)
        consistency = auxiliary_consistency_features(parts, valid_idx, scaler, clf)
        scalar[valid_idx, 3:7] = proto
        scalar[valid_idx, 7:10] = consistency
    return finite_float32(scalar)


def build_reference_scalar_features(reference_parts, target_parts):
    scalar = np.zeros((len(target_parts["df"]), 10), dtype=np.float32)
    scalar[:, :3] = target_parts["agreement"]
    ref_idx = np.arange(len(reference_parts["df"]))
    target_idx = np.arange(len(target_parts["df"]))

    combined = {
        "audio": np.vstack([reference_parts["audio"], target_parts["audio"]]),
        "text": np.vstack([reference_parts["text"], target_parts["text"]]),
        "class_ids": np.concatenate([reference_parts["class_ids"], target_parts["class_ids"]]),
    }
    shifted_target_idx = target_idx + len(reference_parts["df"])
    proto = prototype_features_for_indices(combined, ref_idx, shifted_target_idx)
    scaler, clf = fit_auxiliary_class_model(reference_parts, ref_idx, RUN_CONFIG["seed"])
    consistency = auxiliary_consistency_features(target_parts, target_idx, scaler, clf)
    scalar[:, 3:7] = proto
    scalar[:, 7:10] = consistency
    return finite_float32(scalar)


def class_distribution_report(parts):
    actual = parts["df"]["class"].astype(str).value_counts().sort_index()
    rows = []
    for cls in sorted(set(actual.index.tolist()) | set(SUPPLIED_DATASET_DISTRIBUTION.keys())):
        actual_count = int(actual.get(cls, 0))
        supplied_count = int(SUPPLIED_DATASET_DISTRIBUTION.get(cls, 0))
        rows.append(
            {
                "class": cls,
                "metadata_count": actual_count,
                "supplied_count": supplied_count,
                "diff_metadata_minus_supplied": actual_count - supplied_count,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "v5_class_distribution_mapping.csv", index=False)
    return out


def make_class_sample_weights(parts):
    if not RUN_CONFIG["use_class_sample_weights"]:
        return np.ones(len(parts["df"]), dtype=np.float32)

    classes = parts["df"]["class"].astype(str)
    if RUN_CONFIG["class_weight_source"] == "supplied":
        counts = pd.Series(SUPPLIED_DATASET_DISTRIBUTION, dtype=np.float32)
    elif RUN_CONFIG["class_weight_source"] == "metadata":
        counts = classes.value_counts().astype(np.float32)
    else:
        raise ValueError(f"Unknown class_weight_source: {RUN_CONFIG['class_weight_source']}")

    n_total = float(counts.sum())
    n_classes = float(len(counts))
    weights_by_class = n_total / (n_classes * counts.clip(lower=1.0))
    sample_weights = classes.map(weights_by_class).fillna(1.0).to_numpy(dtype=np.float32)
    sample_weights = sample_weights / max(float(sample_weights.mean()), 1e-8)

    weight_df = (
        pd.DataFrame(
            {
                "class": classes,
                "sample_weight": sample_weights,
            }
        )
        .groupby("class", as_index=False)
        .agg(n=("sample_weight", "size"), sample_weight=("sample_weight", "first"))
        .sort_values("class")
    )
    weight_df.to_csv(REPORT_DIR / "v5_class_sample_weights.csv", index=False)
    return sample_weights.astype(np.float32)


def feature_matrix(parts, scalar_features, feature_set):
    if feature_set == "E1":
        arrays = [parts["audio"], parts["text"], parts["class"]]
    elif feature_set == "E2":
        arrays = [parts["audio"], parts["text"], parts["class"], scalar_features[:, :3]]
    elif feature_set == "E3":
        arrays = [parts["audio"], parts["text"], parts["class"], scalar_features]
    elif feature_set == "E4":
        arrays = [scalar_features, parts["class"]]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return finite_float32(np.hstack(arrays))


class ConfidenceDataset(Dataset):
    def __init__(self, x, y=None, sample_weight=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.sample_weight = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.x[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[idx]
        return item


class OrdinalMLP(nn.Module):
    def __init__(self, input_dim, hidden=(512, 256), dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 4),
        )

    def forward(self, x):
        return self.net(x)


def ordinal_targets(y):
    thresholds = torch.arange(0, 4, device=y.device).view(1, -1)
    return (y.view(-1, 1) > thresholds).float()


def ordinal_loss(logits, y, sample_weight=None):
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    per_threshold = F.binary_cross_entropy_with_logits(logits, ordinal_targets(y), reduction="none")
    per_sample = per_threshold.mean(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    sample_weight = sample_weight.to(per_sample.device)
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def probs_from_ordinal_logits_np(logits):
    logits = np.nan_to_num(
        np.asarray(logits, dtype=np.float32),
        nan=0.0,
        posinf=50.0,
        neginf=-50.0,
    )
    q = 1.0 / (1.0 + np.exp(-logits))
    q = np.minimum.accumulate(q, axis=1)
    q = np.clip(q, 0.0, 1.0)
    probs = np.zeros((q.shape[0], 5), dtype=np.float32)
    probs[:, 0] = 1.0 - q[:, 0]
    probs[:, 1] = q[:, 0] - q[:, 1]
    probs[:, 2] = q[:, 1] - q[:, 2]
    probs[:, 3] = q[:, 2] - q[:, 3]
    probs[:, 4] = q[:, 3]
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs.astype(np.float32)


@torch.no_grad()
def evaluate_mlp(model, loader, y_true_0based=None):
    model.eval()
    losses = []
    logits_all = []
    for batch in loader:
        x = batch["x"].to(DEVICE)
        logits = model(x)
        logits_all.append(logits.cpu().numpy())
        if "y" in batch:
            y = batch["y"].to(DEVICE)
            sample_weight = batch.get("sample_weight")
            if sample_weight is not None:
                sample_weight = sample_weight.to(DEVICE)
            losses.append(float(ordinal_loss(logits, y, sample_weight).item()) * x.shape[0])
    logits_np = np.vstack(logits_all).astype(np.float32)
    probs = probs_from_ordinal_logits_np(logits_np)
    loss = None
    if y_true_0based is not None and losses:
        loss = float(np.sum(losses) / len(y_true_0based))
    return probs, loss


def expected_score_from_probs(probs):
    probs = finite_float32(probs)
    return finite_float32((probs * LABEL_VALUES_NP.reshape(1, -1)).sum(axis=1))


def score_to_class(score):
    return np.clip(np.rint(score), 1, 5).astype(np.int64)


def metrics_from_score(y_true_1based, score, pred_class=None):
    score = finite_float32(score).reshape(-1)
    pred_class = score_to_class(score) if pred_class is None else pred_class.astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_1based,
        pred_class,
        average="macro",
        zero_division=0,
    )
    rho = spearmanr(y_true_1based, score).statistic
    return {
        "mae": float(mean_absolute_error(y_true_1based, score)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "accuracy": float(accuracy_score(y_true_1based, pred_class)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true_1based, pred_class, weights="quadratic")),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def metrics_from_probs(y_true_1based, probs):
    probs = finite_float32(probs)
    score = expected_score_from_probs(probs)
    pred_class = probs.argmax(axis=1) + 1
    return metrics_from_score(y_true_1based, score, pred_class)


def train_mlp_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based, sample_weights):
    seed_everything(RUN_CONFIG["seed"] + fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)
    train_loader = DataLoader(
        ConfidenceDataset(x_train, y_0based[train_idx], sample_weights[train_idx]),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        ConfidenceDataset(x_valid, y_0based[valid_idx], sample_weights[valid_idx]),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=False,
    )

    model = OrdinalMLP(
        x_raw.shape[1],
        hidden=tuple(RUN_CONFIG["hidden"]),
        dropout=RUN_CONFIG["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RUN_CONFIG["learning_rate"],
        weight_decay=RUN_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUN_CONFIG["epochs"])
    best_mae = math.inf
    best_epoch = -1
    best_state = None
    best_probs = None
    stale_epochs = 0
    history = []

    for epoch in range(1, RUN_CONFIG["epochs"] + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            sample_weight = batch["sample_weight"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(x), y, sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        probs, val_loss = evaluate_mlp(model, valid_loader, y_0based[valid_idx])
        val_mae = mean_absolute_error(y_1based[valid_idx], expected_score_from_probs(probs))
        if not np.isfinite(val_mae):
            val_mae = math.inf
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_mae": float(val_mae),
        }
        history.append(row)
        if val_mae < best_mae - 1e-6:
            best_mae = float(val_mae)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_probs = probs
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= RUN_CONFIG["patience"]:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = len(history)
        best_probs, _ = evaluate_mlp(model, valid_loader, y_0based[valid_idx])
        best_mae = float(mean_absolute_error(y_1based[valid_idx], expected_score_from_probs(best_probs)))
        if not np.isfinite(best_mae):
            best_mae = float("inf")
    model.load_state_dict(best_state)
    ckpt_dir = CHECKPOINT_DIR / exp["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "input_dim": int(x_raw.shape[1]),
            "best_epoch": int(best_epoch),
            "experiment": exp,
            "run_config": RUN_CONFIG,
        },
        ckpt_dir / f"fold_{fold}.pt",
    )
    return best_probs, history, {"fold": fold, "best_epoch": int(best_epoch), "best_mae": float(best_mae)}


def fit_xgboost(train_x, train_y, valid_x, valid_y, train_weight, valid_weight, seed):
    model = XGBRegressor(
        n_estimators=RUN_CONFIG["tree_estimators"],
        max_depth=3,
        learning_rate=RUN_CONFIG["tree_learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=RUN_CONFIG["patience"],
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weight,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        sample_weight_eval_set=[train_weight, valid_weight],
        verbose=False,
    )
    evals = model.evals_result()
    train_loss = evals["validation_0"]["mae"]
    val_loss = evals["validation_1"]["mae"]
    history = [
        {"epoch": idx + 1, "train_loss": float(tr), "val_loss": float(va), "val_mae": float(va)}
        for idx, (tr, va) in enumerate(zip(train_loss, val_loss))
    ]
    return model, history


def fit_hist_gradient(train_x, train_y, valid_x, valid_y, train_weight, valid_weight, seed):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=RUN_CONFIG["tree_learning_rate"],
        max_iter=1,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=seed,
        warm_start=True,
    )
    best_model_bytes = None
    best_mae = math.inf
    stale_epochs = 0
    history = []
    for epoch in range(1, RUN_CONFIG["tree_estimators"] + 1):
        model.set_params(max_iter=epoch)
        model.fit(train_x, train_y, sample_weight=train_weight)
        train_pred = finite_float32(np.clip(model.predict(train_x), 1.0, 5.0))
        valid_pred = finite_float32(np.clip(model.predict(valid_x), 1.0, 5.0))
        train_mae = mean_absolute_error(train_y, train_pred)
        val_mae = mean_absolute_error(valid_y, valid_pred)
        if not np.isfinite(val_mae):
            val_mae = math.inf
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_mae),
                "val_loss": float(val_mae),
                "val_mae": float(val_mae),
            }
        )
        if val_mae < best_mae - 1e-6:
            best_mae = float(val_mae)
            best_model_bytes = pickle.dumps(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= RUN_CONFIG["patience"]:
                break
    if best_model_bytes is None:
        best_model_bytes = pickle.dumps(model)
    return pickle.loads(best_model_bytes), history


def train_tree_fold(exp, fold, train_idx, valid_idx, x_raw, y_1based, sample_weights):
    seed = RUN_CONFIG["seed"] + fold
    train_x = x_raw[train_idx]
    valid_x = x_raw[valid_idx]
    train_y = y_1based[train_idx].astype(np.float32)
    valid_y = y_1based[valid_idx].astype(np.float32)
    train_weight = sample_weights[train_idx].astype(np.float32)
    valid_weight = sample_weights[valid_idx].astype(np.float32)
    if XGBRegressor is not None:
        try:
            model, history = fit_xgboost(train_x, train_y, valid_x, valid_y, train_weight, valid_weight, seed)
            backend = "xgboost"
        except Exception as exc:
            print(f"xgboost failed on fold {fold}; falling back to HistGradientBoostingRegressor: {exc}")
            model, history = fit_hist_gradient(train_x, train_y, valid_x, valid_y, train_weight, valid_weight, seed)
            backend = "hist_gradient_boosting"
    else:
        model, history = fit_hist_gradient(train_x, train_y, valid_x, valid_y, train_weight, valid_weight, seed)
        backend = "hist_gradient_boosting"

    score = finite_float32(np.clip(model.predict(valid_x), 1.0, 5.0))
    ckpt_dir = CHECKPOINT_DIR / exp["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / f"fold_{fold}_{backend}.pkl", "wb") as f:
        pickle.dump({"model": model, "backend": backend, "experiment": exp, "run_config": RUN_CONFIG}, f)

    return score, history, {
        "fold": fold,
        "best_epoch": int(history[int(np.argmin([row["val_mae"] for row in history]))]["epoch"]) if history else 0,
        "best_mae": float(min(row["val_mae"] for row in history)) if history else float("inf"),
        "backend": backend,
    }


def run_cv_experiment(exp, parts, scalar_features, y_0based, y_1based, splits, sample_weights):
    x_raw = feature_matrix(parts, scalar_features, exp["feature_set"])
    fold_rows = []
    histories = []
    if exp["model_type"] == "ordinal_mlp":
        oof_probs = np.zeros((len(y_1based), 5), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            probs, history, fold_info = train_mlp_fold(
                exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based, sample_weights
            )
            oof_probs[valid_idx] = probs
            histories.append(history)
            fold_info.update({"experiment": exp["name"], "feature_dim": int(x_raw.shape[1])})
            fold_info.update(metrics_from_probs(y_1based[valid_idx], probs))
            fold_rows.append(fold_info)
            print(f"{exp['name']} fold={fold} mae={fold_info['mae']:.4f} epoch={fold_info['best_epoch']}")
        oof_score = expected_score_from_probs(oof_probs)
        oof_class = oof_probs.argmax(axis=1) + 1
        oof_payload = {"probs": oof_probs, "score": oof_score, "pred_class": oof_class}
    else:
        oof_score = np.zeros(len(y_1based), dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            score, history, fold_info = train_tree_fold(
                exp, fold, train_idx, valid_idx, x_raw, y_1based, sample_weights
            )
            oof_score[valid_idx] = score
            histories.append(history)
            fold_info.update({"experiment": exp["name"], "feature_dim": int(x_raw.shape[1])})
            fold_info.update(metrics_from_score(y_1based[valid_idx], score))
            fold_rows.append(fold_info)
            print(f"{exp['name']} fold={fold} mae={fold_info['mae']:.4f} epoch={fold_info['best_epoch']}")
        oof_payload = {"probs": None, "score": oof_score, "pred_class": score_to_class(oof_score)}

    save_histories(exp["name"], histories)
    save_loss_plot(exp["name"], histories)
    save_confusion_outputs(exp["name"], y_1based, oof_payload["pred_class"])
    return oof_payload, fold_rows, histories, int(x_raw.shape[1])


def save_histories(exp_name, histories):
    for fold, history in enumerate(histories):
        pd.DataFrame(history).to_csv(REPORT_DIR / f"{exp_name}_fold_{fold}_history.csv", index=False)
    aggregate = aggregate_histories(histories)
    aggregate.to_csv(REPORT_DIR / f"{exp_name}_mean_loss_history.csv", index=False)


def aggregate_histories(histories):
    lengths = [len(history) for history in histories]
    if RUN_CONFIG["history_aggregation"] == "truncate":
        target_len = min(lengths)
    else:
        target_len = max(lengths)
    rows = []
    for idx in range(target_len):
        train_vals = []
        val_vals = []
        mae_vals = []
        for history in histories:
            source_idx = min(idx, len(history) - 1)
            train_vals.append(history[source_idx]["train_loss"])
            val_vals.append(history[source_idx]["val_loss"])
            mae_vals.append(history[source_idx]["val_mae"])
        rows.append(
            {
                "epoch": idx + 1,
                "train_loss_mean": float(np.mean(train_vals)),
                "train_loss_std": float(np.std(train_vals)),
                "val_loss_mean": float(np.mean(val_vals)),
                "val_loss_std": float(np.std(val_vals)),
                "val_mae_mean": float(np.mean(mae_vals)),
                "val_mae_std": float(np.std(mae_vals)),
            }
        )
    return pd.DataFrame(rows)


def save_loss_plot(exp_name, histories):
    if plt is None:
        print(f"matplotlib is not available; skipping loss plot for {exp_name}")
        return
    aggregate = aggregate_histories(histories)
    plt.figure(figsize=(8, 4.5))
    plt.plot(aggregate["epoch"], aggregate["train_loss_mean"], label="Train loss")
    plt.plot(aggregate["epoch"], aggregate["val_loss_mean"], label="Val loss")
    plt.fill_between(
        aggregate["epoch"],
        aggregate["train_loss_mean"] - aggregate["train_loss_std"],
        aggregate["train_loss_mean"] + aggregate["train_loss_std"],
        alpha=0.15,
    )
    plt.fill_between(
        aggregate["epoch"],
        aggregate["val_loss_mean"] - aggregate["val_loss_std"],
        aggregate["val_loss_mean"] + aggregate["val_loss_std"],
        alpha=0.15,
    )
    plt.title(f"{exp_name} - Fold-mean loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{exp_name}_mean_loss.png", dpi=160)
    plt.close()


def save_confusion_outputs(exp_name, y_true_1based, pred_class):
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true_1based, pred_class, labels=labels)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]).to_csv(
        REPORT_DIR / f"{exp_name}_confusion_counts.csv"
    )
    pd.DataFrame(cm_norm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]).to_csv(
        REPORT_DIR / f"{exp_name}_confusion_row_normalized.csv"
    )

    if plt is None:
        print(f"matplotlib is not available; skipping confusion plot for {exp_name}")
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.title(f"{exp_name} - Row-normalized OOF confusion")
    plt.xlabel("Predicted confidence")
    plt.ylabel("True confidence")
    plt.xticks(range(5), labels)
    plt.yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, f"{cm_norm[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{exp_name}_confusion_row_normalized.png", dpi=160)
    plt.close()


def fold_summary(fold_df):
    rows = []
    for exp in EXPERIMENTS:
        sub = fold_df[fold_df["experiment"] == exp["name"]]
        row = {
            "experiment": exp["name"],
            "label": exp["label"],
            "feature_dim": int(sub["feature_dim"].iloc[0]),
        }
        for metric in ["mae", "accuracy", "macro_f1", "macro_precision", "macro_recall", "quadratic_weighted_kappa"]:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df, columns=None, float_digits=4):
    view = df.copy() if columns is None else df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [max([len(headers[idx])] + [len(row[idx]) for row in rows]) for idx in range(len(headers))]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def format_mean_std(mean_value, std_value, digits=4, scale=1.0):
    return f"{mean_value * scale:.{digits}f} ± {std_value * scale:.{digits}f}"


def write_report(summary_df, fold_summary_df, fold_df, y_1based, oof_records, elapsed_seconds):
    report_rows = []
    for _, row in fold_summary_df.iterrows():
        report_rows.append(
            {
                "Model": row["label"],
                "Feature Dim": int(row["feature_dim"]),
                "MAE": format_mean_std(row["mae_mean"], row["mae_std"]),
                "Accuracy (%)": format_mean_std(row["accuracy_mean"], row["accuracy_std"], digits=2, scale=100.0),
                "Macro F1": format_mean_std(row["macro_f1_mean"], row["macro_f1_std"]),
                "Macro Precision": format_mean_std(row["macro_precision_mean"], row["macro_precision_std"]),
                "Macro Recall": format_mean_std(row["macro_recall_mean"], row["macro_recall_std"]),
                "QWK": format_mean_std(row["quadratic_weighted_kappa_mean"], row["quadratic_weighted_kappa_std"]),
            }
        )
    report_df = pd.DataFrame(report_rows)

    best = summary_df.sort_values(["mae", "quadratic_weighted_kappa"], ascending=[True, False]).iloc[0]
    lines = [
        "# BSD10k Confidence v5 Experiment Report",
        "",
        f"- Device: `{DEVICE}`",
        f"- Elapsed: {elapsed_seconds / 60:.1f} min",
        f"- Rows: {len(y_1based):,}",
        f"- Folds: {RUN_CONFIG['folds']}",
        f"- MLP architecture: Linear(input, 512) -> GELU -> Dropout -> Linear(512, 256) -> GELU -> Dropout -> Linear(256, 4)",
        f"- Loss: sample-weighted ordinal BCE using metadata `class` inverse-frequency weights",
        f"- Class sample weights: {'enabled' if RUN_CONFIG['use_class_sample_weights'] else 'disabled'}; source=`{RUN_CONFIG['class_weight_source']}`",
        f"- Tree backend: {'XGBoost' if XGBRegressor is not None else 'HistGradientBoostingRegressor'}",
        f"- Plot backend: {'matplotlib' if plt is not None else 'not available; CSVs saved only'}",
        "",
        "## Fold Mean Results",
        "",
        markdown_table(report_df),
        "",
        "## OOF Results",
        "",
        markdown_table(
            summary_df,
            [
                "experiment",
                "feature_dim",
                "mae",
                "accuracy",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "quadratic_weighted_kappa",
                "spearman",
            ],
        ),
        "",
        "## Best Model",
        "",
        f"- Best by MAE: `{best['experiment']}`",
        f"- MAE: {best['mae']:.4f}",
        f"- Accuracy: {best['accuracy'] * 100:.2f}%",
        f"- Macro F1: {best['macro_f1']:.4f}",
        f"- QWK: {best['quadratic_weighted_kappa']:.4f}",
        "",
        "## Saved Artifacts",
        "",
        "- Summary CSV: `outputs/confidence_model_v5/reports/v5_experiment_summary.csv`",
        "- Fold metrics CSV: `outputs/confidence_model_v5/reports/v5_fold_metrics.csv`",
        "- Class distribution mapping CSV: `outputs/confidence_model_v5/reports/v5_class_distribution_mapping.csv`",
        "- Class sample weights CSV: `outputs/confidence_model_v5/reports/v5_class_sample_weights.csv`",
        "- Loss plots: `outputs/confidence_model_v5/plots/*_mean_loss.png`",
        "- Row-normalized OOF confusion plots: `outputs/confidence_model_v5/plots/*_confusion_row_normalized.png`",
        "- OOF predictions: `outputs/confidence_model_v5/predictions/BSD10k_oof_v5_predictions.csv`",
        "",
    ]
    text = "\n".join(lines)
    (ROOT / "confidence_model_v5_report_ko.md").write_text(text, encoding="utf-8")
    (REPORT_DIR / "confidence_model_v5_report_ko.md").write_text(text, encoding="utf-8")


def train_full_mlp(exp, x_raw, y_0based, sample_weights, train_epochs):
    seed_everything(RUN_CONFIG["seed"])
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)
    loader = DataLoader(ConfidenceDataset(x, y_0based, sample_weights), batch_size=RUN_CONFIG["batch_size"], shuffle=True)
    model = OrdinalMLP(x_raw.shape[1], hidden=tuple(RUN_CONFIG["hidden"]), dropout=RUN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_epochs, 1))
    for _ in range(train_epochs):
        model.train()
        for batch in loader:
            x_batch = batch["x"].to(DEVICE)
            y_batch = batch["y"].to(DEVICE)
            sample_weight = batch["sample_weight"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(x_batch), y_batch, sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()
    ckpt_dir = CHECKPOINT_DIR / exp["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "input_dim": int(x_raw.shape[1]),
            "train_epochs": int(train_epochs),
            "experiment": exp,
            "run_config": RUN_CONFIG,
        },
        ckpt_dir / "final_full_bsd10k.pt",
    )
    return model, scaler


def predict_full_mlp(model, scaler, x_raw):
    x = scaler.transform(x_raw).astype(np.float32)
    loader = DataLoader(ConfidenceDataset(x), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
    probs, _ = evaluate_mlp(model, loader)
    return expected_score_from_probs(probs), probs.argmax(axis=1) + 1, probs


def train_full_tree(exp, x_raw, y_1based, sample_weights, train_epochs):
    if XGBRegressor is not None:
        try:
            model = XGBRegressor(
                n_estimators=max(train_epochs, 1),
                max_depth=3,
                learning_rate=RUN_CONFIG["tree_learning_rate"],
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=RUN_CONFIG["seed"],
                n_jobs=-1,
            )
            backend = "xgboost"
            model.fit(x_raw, y_1based.astype(np.float32), sample_weight=sample_weights.astype(np.float32))
        except Exception as exc:
            print(f"xgboost final fit failed; falling back to HistGradientBoostingRegressor: {exc}")
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=RUN_CONFIG["tree_learning_rate"],
                max_iter=max(train_epochs, 1),
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=RUN_CONFIG["seed"],
            )
            backend = "hist_gradient_boosting"
            model.fit(x_raw, y_1based.astype(np.float32), sample_weight=sample_weights.astype(np.float32))
    else:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=RUN_CONFIG["tree_learning_rate"],
            max_iter=max(train_epochs, 1),
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RUN_CONFIG["seed"],
        )
        backend = "hist_gradient_boosting"
        model.fit(x_raw, y_1based.astype(np.float32), sample_weight=sample_weights.astype(np.float32))
    ckpt_dir = CHECKPOINT_DIR / exp["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / f"final_full_bsd10k_{backend}.pkl", "wb") as f:
        pickle.dump({"model": model, "backend": backend, "experiment": exp, "run_config": RUN_CONFIG}, f)
    return model


def save_oof_predictions(parts, y_1based, records):
    out = parts["df"].copy()
    out["true_confidence"] = y_1based
    for exp in EXPERIMENTS:
        record = records[exp["name"]]
        out[f"{exp['name']}_score"] = record["score"]
        out[f"{exp['name']}_class"] = record["pred_class"]
        if record["probs"] is not None:
            for idx in range(5):
                out[f"{exp['name']}_prob_{idx + 1}"] = record["probs"][:, idx]
    out.to_csv(PRED_DIR / "BSD10k_oof_v5_predictions.csv", index=False)


def save_bsd35k_predictions(exp, bsd35k_parts, score, pred_class, probs=None):
    out = bsd35k_parts["df"].copy()
    out["experiment"] = exp["name"]
    out["predicted_confidence_score"] = score
    out["predicted_confidence_class"] = pred_class
    if probs is not None:
        for idx in range(5):
            out[f"prob_confidence_{idx + 1}"] = probs[:, idx]
    out.to_csv(PRED_DIR / f"BSD35k-CS_predicted_v5_{exp['name']}.csv", index=False)


def main():
    start = time.time()
    seed_everything(RUN_CONFIG["seed"])
    print("device:", DEVICE)

    parts, class_categories = load_bsd10k_parts()
    y_1based = parts["df"]["confidence"].to_numpy(dtype=np.int64)
    y_0based = y_1based - 1
    print("BSD10k rows:", len(y_1based), dict(pd.Series(y_1based).value_counts().sort_index()))
    print("class categories:", len(class_categories))
    distribution_df = class_distribution_report(parts)
    if (distribution_df["diff_metadata_minus_supplied"] != 0).any():
        print("supplied class distribution differs from BSD10k_metadata.csv; using metadata counts for sample weights.")
    sample_weights = make_class_sample_weights(parts)
    print(
        "sample weights:",
        {
            "min": float(sample_weights.min()),
            "mean": float(sample_weights.mean()),
            "max": float(sample_weights.max()),
        },
    )

    splitter = StratifiedKFold(n_splits=RUN_CONFIG["folds"], shuffle=True, random_state=RUN_CONFIG["seed"])
    splits = list(splitter.split(np.zeros(len(y_1based)), y_1based))
    scalar_oof = build_oof_scalar_features(parts, splits)
    pd.DataFrame(
        scalar_oof,
        columns=[
            "agreement_cosine",
            "agreement_l2",
            "agreement_dot",
            "proto_audio_label_cos",
            "proto_text_label_cos",
            "proto_audio_margin",
            "proto_text_margin",
            "aux_label_prob",
            "aux_top1_prob",
            "aux_entropy",
        ],
    ).to_csv(REPORT_DIR / "v5_oof_scalar_features.csv", index=False)

    oof_records = {}
    all_fold_rows = []
    all_history = {}
    summary_rows = []

    for exp in EXPERIMENTS:
        payload, fold_rows, histories, feature_dim = run_cv_experiment(
            exp, parts, scalar_oof, y_0based, y_1based, splits, sample_weights
        )
        oof_records[exp["name"]] = payload
        all_fold_rows.extend(fold_rows)
        all_history[exp["name"]] = histories
        metrics = metrics_from_score(y_1based, payload["score"], payload["pred_class"])
        summary_rows.append({"experiment": exp["name"], "feature_dim": feature_dim, **metrics})

    fold_df = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["mae", "quadratic_weighted_kappa"], ascending=[True, False])
    fold_summary_df = fold_summary(fold_df)
    summary_df.to_csv(REPORT_DIR / "v5_experiment_summary.csv", index=False)
    fold_df.to_csv(REPORT_DIR / "v5_fold_metrics.csv", index=False)
    fold_summary_df.to_csv(REPORT_DIR / "v5_fold_mean_std_summary.csv", index=False)
    save_oof_predictions(parts, y_1based, oof_records)

    final_epochs = {
        exp["name"]: int(round(float(fold_df[fold_df["experiment"] == exp["name"]]["best_epoch"].median())))
        for exp in EXPERIMENTS
    }
    final_epochs = {key: max(1, value) for key, value in final_epochs.items()}

    if RUN_CONFIG["train_final_models"]:
        scalar_full = build_reference_scalar_features(parts, parts)
        bsd35k_parts = None
        bsd35k_scalar = None
        if RUN_CONFIG["predict_bsd35k"]:
            bsd35k_parts = load_bsd35k_parts(class_categories)
            bsd35k_scalar = build_reference_scalar_features(parts, bsd35k_parts)

        for exp in EXPERIMENTS:
            x_full = feature_matrix(parts, scalar_full, exp["feature_set"])
            train_epochs = final_epochs[exp["name"]]
            print(f"final train {exp['name']} epochs={train_epochs}")
            if exp["model_type"] == "ordinal_mlp":
                model, scaler = train_full_mlp(exp, x_full, y_0based, sample_weights, train_epochs)
                if bsd35k_parts is not None:
                    x_bsd35k = feature_matrix(bsd35k_parts, bsd35k_scalar, exp["feature_set"])
                    score, pred_class, probs = predict_full_mlp(model, scaler, x_bsd35k)
                    save_bsd35k_predictions(exp, bsd35k_parts, score, pred_class, probs)
            else:
                model = train_full_tree(exp, x_full, y_1based, sample_weights, train_epochs)
                if bsd35k_parts is not None:
                    x_bsd35k = feature_matrix(bsd35k_parts, bsd35k_scalar, exp["feature_set"])
                    score = finite_float32(np.clip(model.predict(x_bsd35k), 1.0, 5.0))
                    save_bsd35k_predictions(exp, bsd35k_parts, score, score_to_class(score))

    elapsed = time.time() - start
    with open(REPORT_DIR / "v5_run_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_config": RUN_CONFIG,
                "device": str(DEVICE),
                "class_categories": class_categories,
                "class_distribution_mapping": distribution_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "elapsed_seconds": elapsed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_report(summary_df, fold_summary_df, fold_df, y_1based, oof_records, elapsed)
    print(summary_df)


if __name__ == "__main__":
    main()
