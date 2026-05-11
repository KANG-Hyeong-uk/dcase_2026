from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEATURE_DIR = DATA_DIR / "features"
OUTPUT_DIR = ROOT / "outputs" / "confidence_model_true_5class"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for path in [REPORT_DIR, PRED_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_VALUES = np.arange(1, 6, dtype=np.float32)

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
    "epochs": 35,
    "patience": 7,
    "batch_size": 256,
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "dropout": 0.25,
    "regression_weight": 0.35,
}

EXPERIMENTS = [
    {"name": "softmax_ce_no_weights", "model_type": "softmax", "feature_set": "baseline_all", "class_weights": False},
    {"name": "softmax_ce_weighted", "model_type": "softmax", "feature_set": "baseline_all", "class_weights": True},
    {"name": "multitask_softmax_reg", "model_type": "multitask_softmax_reg", "feature_set": "baseline_all", "class_weights": False},
    {"name": "ordinal_cumulative", "model_type": "ordinal", "feature_set": "baseline_all", "class_weights": False},
    {"name": "two_tower_softmax", "model_type": "two_tower_softmax", "feature_set": "tower_all", "class_weights": False},
    {"name": "two_tower_multitask", "model_type": "two_tower_multitask", "feature_set": "tower_all", "class_weights": False},
    {"name": "two_tower_ordinal", "model_type": "two_tower_ordinal", "feature_set": "tower_all", "class_weights": False},
]


def clean_metadata(df, require_confidence):
    df = df.copy()
    df["sound_id"] = df["sound_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    if "class_top" not in df.columns:
        df["class_top"] = df["class"].str.split("-").str[0]
    else:
        df["class_top"] = df["class_top"].fillna(df["class"].str.split("-").str[0]).astype(str).str.strip()

    class_idx = df["class_idx"].astype(str).str.strip()
    keep = ~((class_idx.str.len() == 3) & (class_idx.str.endswith("99") | class_idx.str.endswith("00")))
    df = df[keep].copy()

    if require_confidence:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[df["confidence"].isin([1, 2, 3, 4, 5])].copy()
        df["confidence"] = df["confidence"].astype(int)

    return df.reset_index(drop=True)


def filter_rows_with_embeddings(df, audio_dir, text_dir):
    keep = []
    for sound_id in df["sound_id"].astype(str):
        keep.append((audio_dir / f"{sound_id}.npy").is_file() and (text_dir / f"{sound_id}.npy").is_file())
    return df[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def metadata_numeric_features(df):
    title = df.get("title", "").fillna("").astype(str)
    tags = df.get("tags", "").fillna("").astype(str)
    description = df.get("description", "").fillna("").astype(str)

    out = pd.DataFrame(index=df.index)
    out["title_chars"] = title.str.len()
    out["tag_count"] = tags.apply(lambda x: 0 if not x else len([tag for tag in x.split(",") if tag.strip()]))
    out["description_chars"] = description.str.len()
    out["has_description"] = (out["description_chars"] > 0).astype(np.float32)
    return out.to_numpy(dtype=np.float32)


def make_categories(df):
    return (
        sorted(df["class"].astype(str).unique().tolist()),
        sorted(df["class_top"].astype(str).unique().tolist()),
    )


def one_hot(values, categories):
    index = {category: idx for idx, category in enumerate(categories)}
    arr = np.zeros((len(values), len(categories)), dtype=np.float32)
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
            audio_rows.append(np.load(audio_path).reshape(-1).astype(np.float32))
            text_rows.append(np.load(text_path).reshape(-1).astype(np.float32))
            kept.append(idx)
    kept_df = df.reset_index(drop=True).iloc[kept].reset_index(drop=True).copy()
    return kept_df, np.vstack(audio_rows).astype(np.float32), np.vstack(text_rows).astype(np.float32)


def build_parts(df, audio_dir, text_dir, class_categories, top_class_categories):
    kept_df, audio, text = load_embeddings(df, audio_dir, text_dir)
    class_x = one_hot(kept_df["class"].astype(str).tolist(), class_categories)
    top_x = one_hot(kept_df["class_top"].astype(str).tolist(), top_class_categories)
    meta = metadata_numeric_features(kept_df)
    diff = np.abs(audio - text).astype(np.float32)
    product = (audio * text).astype(np.float32)
    dot = np.sum(product, axis=1, keepdims=True).astype(np.float32)
    audio_norm = np.linalg.norm(audio, axis=1, keepdims=True)
    text_norm = np.linalg.norm(text, axis=1, keepdims=True)
    cosine = (dot / np.maximum(audio_norm * text_norm, 1e-8)).astype(np.float32)
    l2 = np.linalg.norm(audio - text, axis=1, keepdims=True).astype(np.float32)
    return {
        "df": kept_df,
        "audio": audio,
        "text": text,
        "class": class_x,
        "top_class": top_x,
        "meta": meta,
        "diff": diff,
        "product": product,
        "dot": dot,
        "cosine": cosine,
        "l2": l2,
    }


def feature_matrix(parts, feature_set):
    if feature_set == "baseline_all":
        arrays = [parts["audio"], parts["text"], parts["class"], parts["top_class"], parts["meta"]]
    elif feature_set == "interaction_all":
        arrays = [
            parts["audio"],
            parts["text"],
            parts["diff"],
            parts["product"],
            parts["cosine"],
            parts["dot"],
            parts["l2"],
            parts["class"],
            parts["top_class"],
            parts["meta"],
        ]
    elif feature_set == "embedding_meta_only":
        arrays = [parts["audio"], parts["text"], parts["meta"]]
    elif feature_set == "prior_only":
        arrays = [parts["class"], parts["top_class"], parts["meta"]]
    elif feature_set == "interaction_no_class":
        arrays = [
            parts["audio"],
            parts["text"],
            parts["diff"],
            parts["product"],
            parts["cosine"],
            parts["dot"],
            parts["l2"],
            parts["meta"],
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.hstack(arrays).astype(np.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ConfidenceDataset(Dataset):
    def __init__(self, x, y_class=None, y_score=None, y_ordinal=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_class = None if y_class is None else torch.tensor(y_class, dtype=torch.long)
        self.y_score = None if y_score is None else torch.tensor(y_score, dtype=torch.float32)
        self.y_ordinal = None if y_ordinal is None else torch.tensor(y_ordinal, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.x[idx]}
        if self.y_class is not None:
            item["y_class"] = self.y_class[idx]
        if self.y_score is not None:
            item["y_score"] = self.y_score[idx]
        if self.y_ordinal is not None:
            item["y_ordinal"] = self.y_ordinal[idx]
        return item


class MLPBackbone(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SoftmaxMLP(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = MLPBackbone(input_dim, dropout)
        self.head = nn.Linear(128, 5)

    def forward(self, x):
        return self.head(self.backbone(x))


class MultiTaskSoftmaxRegMLP(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = MLPBackbone(input_dim, dropout)
        self.class_head = nn.Linear(128, 5)
        self.score_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.class_head(h), self.score_head(h).squeeze(1)


class OrdinalMLP(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = MLPBackbone(input_dim, dropout)
        self.head = nn.Linear(128, 4)

    def forward(self, x):
        return self.head(self.backbone(x))


class TwoTowerBackbone(nn.Module):
    def __init__(self, tab_dim, dropout):
        super().__init__()
        self.audio_tower = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.text_tower = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 * 4 + tab_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        audio = x[:, :512]
        text = x[:, 512:1024]
        tab = x[:, 1024:]
        audio_z = self.audio_tower(audio)
        text_z = self.text_tower(text)
        fused = torch.cat([audio_z, text_z, torch.abs(audio_z - text_z), audio_z * text_z, tab], dim=1)
        return self.fusion(fused)


class TwoTowerSoftmax(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = TwoTowerBackbone(input_dim - 1024, dropout)
        self.head = nn.Linear(128, 5)

    def forward(self, x):
        return self.head(self.backbone(x))


class TwoTowerMultiTask(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = TwoTowerBackbone(input_dim - 1024, dropout)
        self.class_head = nn.Linear(128, 5)
        self.score_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.class_head(h), self.score_head(h).squeeze(1)


class TwoTowerOrdinal(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.backbone = TwoTowerBackbone(input_dim - 1024, dropout)
        self.head = nn.Linear(128, 4)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_tower_matrix(parts):
    scalar = np.hstack([parts["cosine"], parts["dot"], parts["l2"]]).astype(np.float32)
    tab = np.hstack([parts["class"], parts["top_class"], parts["meta"], scalar]).astype(np.float32)
    return np.hstack([parts["audio"], parts["text"], tab]).astype(np.float32)


def matrix_for_feature_set(parts, feature_set):
    if feature_set == "baseline_all":
        return feature_matrix(parts, "baseline_all")
    if feature_set == "tower_all":
        return build_tower_matrix(parts)
    raise ValueError(feature_set)


def make_model(exp, input_dim):
    if exp["model_type"] == "softmax":
        return SoftmaxMLP(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    if exp["model_type"] == "multitask_softmax_reg":
        return MultiTaskSoftmaxRegMLP(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    if exp["model_type"] == "ordinal":
        return OrdinalMLP(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    if exp["model_type"] == "two_tower_softmax":
        return TwoTowerSoftmax(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    if exp["model_type"] == "two_tower_multitask":
        return TwoTowerMultiTask(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    if exp["model_type"] == "two_tower_ordinal":
        return TwoTowerOrdinal(input_dim, RUN_CONFIG["dropout"]).to(DEVICE)
    raise ValueError(exp["model_type"])


def ordinal_probs_from_logits(logits):
    q = torch.sigmoid(logits)
    # Enforce monotonic cumulative probabilities q2 >= q3 >= q4 >= q5.
    q_np = q.detach().cpu().numpy()
    q_np = np.minimum.accumulate(q_np, axis=1)
    q_np = np.clip(q_np, 0.0, 1.0)
    probs = np.zeros((q_np.shape[0], 5), dtype=np.float32)
    probs[:, 0] = 1.0 - q_np[:, 0]
    probs[:, 1] = q_np[:, 0] - q_np[:, 1]
    probs[:, 2] = q_np[:, 1] - q_np[:, 2]
    probs[:, 3] = q_np[:, 2] - q_np[:, 3]
    probs[:, 4] = q_np[:, 3]
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    return probs


def predict_model(model, loader, model_type):
    model.eval()
    all_probs = []
    all_score = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            if model_type in {"softmax", "two_tower_softmax"}:
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
            elif model_type in {"multitask_softmax_reg", "two_tower_multitask"}:
                logits, score_logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                score = (torch.sigmoid(score_logits).cpu().numpy() * 4.0) + 1.0
                all_probs.append(probs)
                all_score.append(score)
            elif model_type in {"ordinal", "two_tower_ordinal"}:
                logits = model(x)
                probs = ordinal_probs_from_logits(logits)
                all_probs.append(probs)
    probs = np.vstack(all_probs)
    if all_score:
        return probs, np.concatenate(all_score)
    return probs, None


def expected_score(probs):
    return probs @ LABEL_VALUES


def metrics_from_probs(y_true_1based, probs, score_override=None):
    pred_class = probs.argmax(axis=1) + 1
    pred_score = expected_score(probs) if score_override is None else score_override
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_1based,
        pred_class,
        labels=[1, 2, 3, 4, 5],
        average="macro",
        zero_division=0,
    )
    return {
        "mae_expected": float(mean_absolute_error(y_true_1based, expected_score(probs))),
        "mae_score": float(mean_absolute_error(y_true_1based, pred_score)),
        "accuracy": float(accuracy_score(y_true_1based, pred_class)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true_1based, pred_class, weights="quadratic")),
    }


def class_weights(labels_0based):
    counts = np.bincount(labels_0based, minlength=5).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based, y_score_0to1, y_ordinal):
    ckpt_path = CHECKPOINT_DIR / exp["name"] / f"fold_{fold}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)
        valid_loader = DataLoader(ConfidenceDataset(x_valid), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
        model = make_model(exp, ckpt["input_dim"])
        model.load_state_dict(ckpt["model_state"])
        probs, score_override = predict_model(model, valid_loader, exp["model_type"])
        metrics = metrics_from_probs(y_1based[valid_idx], probs, score_override)
        return probs, score_override, {
            "fold": fold,
            "best_epoch": ckpt["best_epoch"],
            "best_mae": float(metrics["mae_score"]),
            "reused_checkpoint": True,
        }

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)
    train_ds = ConfidenceDataset(
        x_train,
        y_class=y_0based[train_idx],
        y_score=y_score_0to1[train_idx],
        y_ordinal=y_ordinal[train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=RUN_CONFIG["batch_size"], shuffle=True)
    valid_loader = DataLoader(ConfidenceDataset(x_valid), batch_size=RUN_CONFIG["batch_size"], shuffle=False)

    model = make_model(exp, x_train.shape[1])
    ce = nn.CrossEntropyLoss(weight=class_weights(y_0based[train_idx]).to(DEVICE)) if exp["class_weights"] else nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    reg = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=RUN_CONFIG["learning_rate"],
        weight_decay=RUN_CONFIG["weight_decay"],
    )

    best_mae = np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, RUN_CONFIG["epochs"] + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y_class = batch["y_class"].to(DEVICE)
            y_score = batch["y_score"].to(DEVICE)
            y_ord = batch["y_ordinal"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            if exp["model_type"] in {"softmax", "two_tower_softmax"}:
                logits = model(x)
                loss = ce(logits, y_class)
            elif exp["model_type"] in {"multitask_softmax_reg", "two_tower_multitask"}:
                logits, score_logits = model(x)
                loss = ce(logits, y_class) + RUN_CONFIG["regression_weight"] * reg(torch.sigmoid(score_logits), y_score)
            elif exp["model_type"] in {"ordinal", "two_tower_ordinal"}:
                logits = model(x)
                loss = bce(logits, y_ord)
            else:
                raise ValueError(exp["model_type"])
            loss.backward()
            optimizer.step()

        probs, score_override = predict_model(model, valid_loader, exp["model_type"])
        score_for_mae = score_override if score_override is not None else None
        metrics = metrics_from_probs(y_1based[valid_idx], probs, score_for_mae)
        monitor = metrics["mae_score"]
        if monitor < best_mae:
            best_mae = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= RUN_CONFIG["patience"]:
            break

    model.load_state_dict(best_state)
    probs, score_override = predict_model(model, valid_loader, exp["model_type"])
    exp_dir = CHECKPOINT_DIR / exp["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": exp,
            "input_dim": int(x_train.shape[1]),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "model_state": best_state,
            "best_epoch": best_epoch,
            "fold": fold,
        },
        exp_dir / f"fold_{fold}.pt",
    )
    return probs, score_override, {"fold": fold, "best_epoch": best_epoch, "best_mae": float(best_mae)}


def load_bsd10k_parts():
    df = clean_metadata(pd.read_csv(CONFIG["bsd10k_metadata"]), require_confidence=True)
    df = filter_rows_with_embeddings(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"])
    class_categories, top_class_categories = make_categories(df)
    return build_parts(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"], class_categories, top_class_categories)


def load_bsd35k_parts():
    train_df = clean_metadata(pd.read_csv(CONFIG["bsd10k_metadata"]), require_confidence=True)
    train_df = filter_rows_with_embeddings(train_df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"])
    class_categories, top_class_categories = make_categories(train_df)
    df = clean_metadata(pd.read_csv(CONFIG["bsd35k_metadata"]), require_confidence=False)
    df = filter_rows_with_embeddings(df, CONFIG["bsd35k_audio_dir"], CONFIG["bsd35k_text_dir"])
    return build_parts(df, CONFIG["bsd35k_audio_dir"], CONFIG["bsd35k_text_dir"], class_categories, top_class_categories)


def cross_fit_score_ensemble(y_true, pred_score_map):
    names = list(pred_score_map)
    x = np.column_stack([pred_score_map[name] for name in names])
    oof = np.zeros(len(y_true), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=RUN_CONFIG["folds"], shuffle=True, random_state=RUN_CONFIG["seed"])
    for train_idx, valid_idx in splitter.split(x, y_true):
        model = Ridge(alpha=1.0)
        model.fit(x[train_idx], y_true[train_idx])
        oof[valid_idx] = model.predict(x[valid_idx])
    model = Ridge(alpha=1.0)
    model.fit(x, y_true)
    return np.clip(oof, 1.0, 5.0), model, names


def cross_fit_class_stacker(y_true_0based, prob_map):
    names = list(prob_map)
    x = np.hstack([prob_map[name] for name in names])
    oof_probs = np.zeros((len(y_true_0based), 5), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=RUN_CONFIG["folds"], shuffle=True, random_state=RUN_CONFIG["seed"])
    for train_idx, valid_idx in splitter.split(x, y_true_0based):
        model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        model.fit(x[train_idx], y_true_0based[train_idx])
        oof_probs[valid_idx] = model.predict_proba(x[valid_idx])
    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    model.fit(x, y_true_0based)
    return oof_probs, model, names


def predict_bsd35k_experiment(exp, parts):
    x_raw = matrix_for_feature_set(parts, exp["feature_set"])
    probs_all = []
    scores_all = []
    for ckpt_path in sorted((CHECKPOINT_DIR / exp["name"]).glob("fold_*.pt")):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"]
        scaler.scale_ = ckpt["scaler_scale"]
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        x = scaler.transform(x_raw).astype(np.float32)
        loader = DataLoader(ConfidenceDataset(x), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
        model = make_model(exp, ckpt["input_dim"])
        model.load_state_dict(ckpt["model_state"])
        probs, scores = predict_model(model, loader, exp["model_type"])
        probs_all.append(probs)
        if scores is not None:
            scores_all.append(scores)
    probs_mean = np.mean(np.stack(probs_all, axis=0), axis=0)
    score_mean = np.mean(np.stack(scores_all, axis=0), axis=0) if scores_all else None
    return probs_mean, score_mean


def main():
    seed_everything(RUN_CONFIG["seed"])
    start = time.time()
    print("device:", DEVICE)
    parts = load_bsd10k_parts()
    y_1based = parts["df"]["confidence"].to_numpy(dtype=np.int64)
    y_0based = y_1based - 1
    y_score_0to1 = ((y_1based.astype(np.float32) - 1.0) / 4.0).astype(np.float32)
    y_ordinal = np.column_stack([(y_1based >= threshold).astype(np.float32) for threshold in [2, 3, 4, 5]])
    print("rows:", len(y_1based), "distribution:", dict(pd.Series(y_1based).value_counts().sort_index()))

    splitter = StratifiedKFold(n_splits=RUN_CONFIG["folds"], shuffle=True, random_state=RUN_CONFIG["seed"])
    splits = list(splitter.split(np.zeros(len(y_1based)), y_1based))

    summary_rows = []
    fold_rows = []
    oof_frames = []
    prob_map = {}
    score_map = {}

    for exp in EXPERIMENTS:
        seed_everything(RUN_CONFIG["seed"])
        x_raw = matrix_for_feature_set(parts, exp["feature_set"])
        print("experiment:", exp["name"], "feature_dim:", x_raw.shape[1])
        oof_probs = np.zeros((len(y_1based), 5), dtype=np.float32)
        oof_score_override = np.full(len(y_1based), np.nan, dtype=np.float32)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            probs, score_override, fold_info = train_one_fold(
                exp,
                fold,
                train_idx,
                valid_idx,
                x_raw,
                y_0based,
                y_1based,
                y_score_0to1,
                y_ordinal,
            )
            oof_probs[valid_idx] = probs
            if score_override is not None:
                oof_score_override[valid_idx] = score_override
            fold_info.update({"experiment": exp["name"], "model_type": exp["model_type"], "feature_set": exp["feature_set"]})
            fold_rows.append(fold_info)
            print(" ", fold, "mae", round(fold_info["best_mae"], 4), "best_epoch", fold_info["best_epoch"])

        score_override = None if np.isnan(oof_score_override).all() else oof_score_override
        row = {
            "experiment": exp["name"],
            "model_type": exp["model_type"],
            "feature_set": exp["feature_set"],
            "feature_dim": int(x_raw.shape[1]),
            **metrics_from_probs(y_1based, oof_probs, score_override),
        }
        summary_rows.append(row)
        prob_map[exp["name"]] = oof_probs
        score_map[exp["name"]] = score_override if score_override is not None else expected_score(oof_probs)

        frame = parts["df"].copy()
        frame["experiment"] = exp["name"]
        frame["true_confidence"] = y_1based
        frame["predicted_confidence_class"] = oof_probs.argmax(axis=1) + 1
        frame["predicted_confidence_score"] = expected_score(oof_probs)
        if score_override is not None:
            frame["predicted_confidence_score_regression_head"] = score_override
        for i in range(5):
            frame[f"prob_confidence_{i + 1}"] = oof_probs[:, i]
        oof_frames.append(frame)

    ensemble_defs = {
        "ensemble_avg_softmax_ordinal_tower": [
            "softmax_ce_no_weights",
            "multitask_softmax_reg",
            "ordinal_cumulative",
            "two_tower_multitask",
            "two_tower_ordinal",
        ],
        "ensemble_avg_best_ordinal_towers": [
            "ordinal_cumulative",
            "two_tower_multitask",
            "two_tower_ordinal",
        ],
    }
    for name, members in ensemble_defs.items():
        if all(member in prob_map for member in members):
            probs = np.mean(np.stack([prob_map[m] for m in members], axis=0), axis=0)
            summary_rows.append(
                {
                    "experiment": name,
                    "model_type": "ensemble_average",
                    "feature_set": "+".join(members),
                    "feature_dim": len(members),
                    **metrics_from_probs(y_1based, probs),
                }
            )
            prob_map[name] = probs
            score_map[name] = expected_score(probs)

    stack_probs, stack_model, stack_names = cross_fit_class_stacker(y_0based, {k: prob_map[k] for k in list(prob_map) if not k.startswith("ensemble_")})
    prob_map["stacked_class_logistic"] = stack_probs
    score_map["stacked_class_logistic"] = expected_score(stack_probs)
    summary_rows.append(
        {
            "experiment": "stacked_class_logistic",
            "model_type": "ensemble_stacking",
            "feature_set": "+".join(stack_names),
            "feature_dim": len(stack_names),
            **metrics_from_probs(y_1based, stack_probs),
        }
    )

    score_ensemble, score_model, score_names = cross_fit_score_ensemble(y_1based.astype(np.float32), score_map)
    # Convert score-only ensemble to narrow Gaussian-like class probabilities for class metrics.
    distances = np.abs(score_ensemble.reshape(-1, 1) - LABEL_VALUES.reshape(1, -1))
    score_probs = np.exp(-distances)
    score_probs = score_probs / score_probs.sum(axis=1, keepdims=True)
    summary_rows.append(
        {
            "experiment": "stacked_score_ridge",
            "model_type": "score_stacking",
            "feature_set": "+".join(score_names),
            "feature_dim": len(score_names),
            **metrics_from_probs(y_1based, score_probs, score_ensemble),
        }
    )

    summary = pd.DataFrame(summary_rows).sort_values(["mae_score", "quadratic_weighted_kappa"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    oof_all = pd.concat(oof_frames, ignore_index=True)
    summary.to_csv(REPORT_DIR / "true_5class_experiment_summary.csv", index=False)
    folds.to_csv(REPORT_DIR / "true_5class_fold_metrics.csv", index=False)
    oof_all.to_csv(PRED_DIR / "BSD10k_oof_true_5class_experiments.csv", index=False)

    best_name = summary.iloc[0]["experiment"]
    best_probs = prob_map.get(best_name, score_probs if best_name == "stacked_score_ridge" else None)
    best_score = score_map.get(best_name, score_ensemble if best_name == "stacked_score_ridge" else expected_score(best_probs))
    cm = pd.DataFrame(
        confusion_matrix(y_1based, best_probs.argmax(axis=1) + 1, labels=[1, 2, 3, 4, 5]),
        index=[f"true_{i}" for i in range(1, 6)],
        columns=[f"pred_{i}" for i in range(1, 6)],
    )
    cm.to_csv(REPORT_DIR / "best_true_5class_confusion_matrix.csv")

    pred_dist = pd.Series(best_probs.argmax(axis=1) + 1).value_counts().sort_index().rename_axis("predicted_confidence_class").reset_index(name="n")
    pred_dist["rate"] = pred_dist["n"] / pred_dist["n"].sum()
    pred_dist.to_csv(REPORT_DIR / "best_true_5class_prediction_distribution.csv", index=False)

    # BSD35k prediction for best practical probability ensemble. If best is score-only, use score probabilities.
    bsd35k_parts = load_bsd35k_parts()
    bsd35k_prob_map = {}
    bsd35k_score_map = {}
    for exp in EXPERIMENTS:
        probs, score_override = predict_bsd35k_experiment(exp, bsd35k_parts)
        bsd35k_prob_map[exp["name"]] = probs
        bsd35k_score_map[exp["name"]] = score_override if score_override is not None else expected_score(probs)

    for name, members in ensemble_defs.items():
        if all(member in bsd35k_prob_map for member in members):
            bsd35k_prob_map[name] = np.mean(np.stack([bsd35k_prob_map[m] for m in members], axis=0), axis=0)
            bsd35k_score_map[name] = expected_score(bsd35k_prob_map[name])

    if best_name == "stacked_class_logistic":
        x = np.hstack([bsd35k_prob_map[name] for name in stack_names])
        best_bsd35k_probs = stack_model.predict_proba(x)
        best_bsd35k_score = expected_score(best_bsd35k_probs)
    elif best_name == "stacked_score_ridge":
        x = np.column_stack([bsd35k_score_map[name] for name in score_names])
        best_bsd35k_score = np.clip(score_model.predict(x), 1.0, 5.0)
        distances = np.abs(best_bsd35k_score.reshape(-1, 1) - LABEL_VALUES.reshape(1, -1))
        best_bsd35k_probs = np.exp(-distances)
        best_bsd35k_probs = best_bsd35k_probs / best_bsd35k_probs.sum(axis=1, keepdims=True)
    else:
        best_bsd35k_probs = bsd35k_prob_map[best_name]
        best_bsd35k_score = bsd35k_score_map[best_name]

    bsd35k_out = bsd35k_parts["df"].copy()
    bsd35k_out["best_experiment"] = best_name
    bsd35k_out["predicted_confidence_class"] = best_bsd35k_probs.argmax(axis=1) + 1
    bsd35k_out["predicted_confidence_score"] = best_bsd35k_score
    for i in range(5):
        bsd35k_out[f"prob_confidence_{i + 1}"] = best_bsd35k_probs[:, i]
    bsd35k_out.to_csv(PRED_DIR / f"BSD35k-CS_predicted_true_5class_{best_name}.csv", index=False)

    bsd35k_summary = {
        "best_experiment": best_name,
        "bsd35k_rows": int(len(bsd35k_out)),
        "mean_predicted_confidence_score": float(np.mean(best_bsd35k_score)),
        "predicted_class_distribution": {
            str(k): int(v)
            for k, v in pd.Series(bsd35k_out["predicted_confidence_class"]).value_counts().sort_index().items()
        },
        "prediction_file": str(PRED_DIR / f"BSD35k-CS_predicted_true_5class_{best_name}.csv"),
        "elapsed_seconds": time.time() - start,
        "stack_class_members": stack_names,
        "stack_score_members": score_names,
    }
    with open(REPORT_DIR / "true_5class_experiment_report.json", "w", encoding="utf-8") as f:
        json.dump({"run_config": RUN_CONFIG, "best": summary.iloc[0].to_dict(), "bsd35k": bsd35k_summary}, f, indent=2, ensure_ascii=False)

    print(summary)
    print("best:", best_name)
    print(json.dumps(bsd35k_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
