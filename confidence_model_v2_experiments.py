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


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
FEATURE_DIR = DATA_DIR / "features"
OUTPUT_DIR = ROOT / "outputs" / "confidence_model_v2"
REPORT_DIR = OUTPUT_DIR / "reports"
PRED_DIR = OUTPUT_DIR / "predictions"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for path in [REPORT_DIR, PRED_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_VALUES = torch.arange(1, 6, dtype=torch.float32, device=DEVICE)
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
    "prior_sampling_repeats": 500,
}

STAGE1_EXPERIMENTS = [
    {"name": "A_baseline_ce", "stage": "stage1", "feature_set": "baseline", "loss": "ce"},
    {"name": "B_similarity_ce", "stage": "stage1", "feature_set": "similarity", "loss": "ce"},
    {"name": "C_clap_stats_ce", "stage": "stage1", "feature_set": "clap_stats", "loss": "ce"},
    {"name": "D_everything_ce", "stage": "stage1", "feature_set": "everything", "loss": "ce"},
]

STAGE2_LOSSES = ["ce", "ordinal_smoothing", "emd", "expected_mse_aux"]


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ConfidenceDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.x[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


class ConfidenceMLP(nn.Module):
    def __init__(self, input_dim, n_classes=5, hidden=(512, 256), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, x):
        return self.net(x)


def normalized_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def class_text_prototypes(train_parts, class_categories):
    text_norm = normalized_rows(train_parts["text"])
    prototypes = []
    cls = train_parts["df"]["class"].astype(str).to_numpy()
    for category in class_categories:
        mask = cls == category
        if mask.any():
            proto = text_norm[mask].mean(axis=0)
        else:
            proto = np.zeros(text_norm.shape[1], dtype=np.float32)
        proto = proto / max(np.linalg.norm(proto), 1e-8)
        prototypes.append(proto.astype(np.float32))
    return np.vstack(prototypes).astype(np.float32)


def clap_class_stats(audio, prototypes):
    audio_norm = normalized_rows(audio)
    scores = audio_norm @ prototypes.T
    top2 = np.sort(scores, axis=1)[:, -2:]
    top1_score = top2[:, 1:2]
    top2_score = top2[:, 0:1]
    gap = top1_score - top2_score
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=1, keepdims=True)
    return {
        "clap_top1_score": top1_score.astype(np.float32),
        "clap_top2_score": top2_score.astype(np.float32),
        "clap_score_gap": gap.astype(np.float32),
        "clap_score_entropy": entropy.astype(np.float32),
    }


def frequency_features(df, train_df):
    class_counts = train_df["class"].astype(str).value_counts()
    top_counts = train_df["class_top"].astype(str).value_counts()
    class_freq = df["class"].astype(str).map(class_counts).fillna(0).to_numpy(dtype=np.float32)
    top_freq = df["class_top"].astype(str).map(top_counts).fillna(0).to_numpy(dtype=np.float32)
    return np.column_stack([np.log1p(class_freq), np.log1p(top_freq)]).astype(np.float32)


def build_parts(df, audio_dir, text_dir, class_categories, top_class_categories, train_parts=None):
    kept_df, audio, text = load_embeddings(df, audio_dir, text_dir)
    class_x = one_hot(kept_df["class"].astype(str).tolist(), class_categories)
    top_x = one_hot(kept_df["class_top"].astype(str).tolist(), top_class_categories)
    meta = metadata_numeric_features(kept_df)

    product = (audio * text).astype(np.float32)
    abs_diff = np.abs(audio - text).astype(np.float32)
    dot = np.sum(product, axis=1, keepdims=True).astype(np.float32)
    audio_norm = np.linalg.norm(audio, axis=1, keepdims=True).astype(np.float32)
    text_norm = np.linalg.norm(text, axis=1, keepdims=True).astype(np.float32)
    cos_sim = (dot / np.maximum(audio_norm * text_norm, 1e-8)).astype(np.float32)
    l2_dist = np.linalg.norm(audio - text, axis=1, keepdims=True).astype(np.float32)

    reference_parts = train_parts
    prototypes = class_text_prototypes(reference_parts, class_categories)
    clap_stats = clap_class_stats(audio, prototypes)
    freq = frequency_features(kept_df, reference_parts["df"])
    summary = np.column_stack(
        [
            abs_diff.mean(axis=1),
            abs_diff.std(axis=1),
            abs_diff.max(axis=1),
            product.mean(axis=1),
            product.std(axis=1),
        ]
    ).astype(np.float32)

    return {
        "df": kept_df,
        "audio": audio,
        "text": text,
        "class": class_x,
        "top_class": top_x,
        "meta": meta,
        "cos_sim_audio_text": cos_sim,
        "l2_dist_audio_text": l2_dist,
        "audio_emb_norm": audio_norm,
        "text_emb_norm": text_norm,
        "clap_top1_score": clap_stats["clap_top1_score"],
        "clap_top2_score": clap_stats["clap_top2_score"],
        "clap_score_gap": clap_stats["clap_score_gap"],
        "clap_score_entropy": clap_stats["clap_score_entropy"],
        "dot_audio_text": dot,
        "abs_diff": abs_diff,
        "product": product,
        "summary": summary,
        "freq": freq,
    }


def matrix_for_feature_set(parts, feature_set):
    baseline = [parts["audio"], parts["text"], parts["class"], parts["top_class"], parts["meta"]]
    similarity = [
        parts["cos_sim_audio_text"],
        parts["l2_dist_audio_text"],
        parts["audio_emb_norm"],
        parts["text_emb_norm"],
    ]
    clap_stats = [
        parts["clap_top1_score"],
        parts["clap_top2_score"],
        parts["clap_score_gap"],
        parts["clap_score_entropy"],
    ]
    everything_extra = [parts["dot_audio_text"], parts["summary"], parts["freq"], parts["abs_diff"], parts["product"]]

    if feature_set == "baseline":
        arrays = baseline
    elif feature_set == "similarity":
        arrays = baseline + similarity
    elif feature_set == "clap_stats":
        arrays = baseline + similarity + clap_stats
    elif feature_set == "everything":
        arrays = baseline + similarity + clap_stats + everything_extra
    else:
        raise ValueError(feature_set)
    return np.hstack(arrays).astype(np.float32)


def load_bsd10k_parts():
    df = pd.read_csv(CONFIG["bsd10k_metadata"])
    df = clean_metadata(df, require_confidence=True)
    class_categories, top_class_categories = make_categories(df)
    base_parts = build_parts_for_reference(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"], class_categories, top_class_categories)
    parts = build_parts(df, CONFIG["bsd10k_audio_dir"], CONFIG["bsd10k_text_dir"], class_categories, top_class_categories, base_parts)
    return parts, class_categories, top_class_categories


def build_parts_for_reference(df, audio_dir, text_dir, class_categories, top_class_categories):
    kept_df, audio, text = load_embeddings(df, audio_dir, text_dir)
    return {
        "df": kept_df,
        "audio": audio,
        "text": text,
        "class": one_hot(kept_df["class"].astype(str).tolist(), class_categories),
        "top_class": one_hot(kept_df["class_top"].astype(str).tolist(), top_class_categories),
        "meta": metadata_numeric_features(kept_df),
    }


def load_bsd35k_parts(class_categories, top_class_categories, train_parts):
    df = pd.read_csv(CONFIG["bsd35k_metadata"])
    df = clean_metadata(df, require_confidence=False)
    return build_parts(df, CONFIG["bsd35k_audio_dir"], CONFIG["bsd35k_text_dir"], class_categories, top_class_categories, train_parts)


def expected_score_from_probs(probs):
    return (probs * LABEL_VALUES_NP.reshape(1, -1)).sum(axis=1)


def metrics_from_probs(y_true_1based, probs):
    score = expected_score_from_probs(probs)
    pred_class = probs.argmax(axis=1) + 1
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


def ordinal_soft_targets(y, n_classes=5, neighbor=0.05):
    target = torch.zeros((y.shape[0], n_classes), dtype=torch.float32, device=y.device)
    for row, cls in enumerate(y.tolist()):
        target[row, cls] = 1.0 - neighbor
        if cls == 0:
            target[row, 1] = neighbor
        elif cls == n_classes - 1:
            target[row, n_classes - 2] = neighbor
        else:
            target[row, cls - 1] = neighbor
            target[row, cls + 1] = neighbor
            target[row, cls] = 1.0 - 2 * neighbor
    return target


def compute_loss(logits, y, loss_name):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)

    probs = F.softmax(logits, dim=1)
    if loss_name == "ordinal_smoothing":
        target = ordinal_soft_targets(y)
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    one_hot = F.one_hot(y, num_classes=5).float()
    if loss_name == "emd":
        pred_cdf = probs.cumsum(dim=1)
        true_cdf = one_hot.cumsum(dim=1)
        return torch.mean((pred_cdf - true_cdf) ** 2)

    if loss_name == "expected_mse_aux":
        expected = (probs * LABEL_VALUES).sum(dim=1)
        target_score = y.float() + 1.0
        return F.cross_entropy(logits, y) + RUN_CONFIG["expected_mse_weight"] * F.mse_loss(expected, target_score)

    raise ValueError(loss_name)


def predict_probs(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            probs.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.vstack(probs).astype(np.float32)


def train_one_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based):
    seed_everything(RUN_CONFIG["seed"] + fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_valid = scaler.transform(x_raw[valid_idx]).astype(np.float32)

    train_loader = DataLoader(
        ConfidenceDataset(x_train, y_0based[train_idx]),
        batch_size=RUN_CONFIG["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(ConfidenceDataset(x_valid), batch_size=RUN_CONFIG["batch_size"], shuffle=False)

    model = ConfidenceMLP(
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
    stale_epochs = 0
    history = []

    for epoch in range(RUN_CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(x), y, exp["loss"])
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        probs = predict_probs(model, valid_loader)
        val_mae = mean_absolute_error(y_1based[valid_idx], expected_score_from_probs(probs))
        history.append({"epoch": epoch, "train_loss": train_loss / max(seen, 1), "val_mae": float(val_mae)})
        if val_mae < best_mae - 1e-6:
            best_mae = float(val_mae)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= RUN_CONFIG["patience"]:
                break

    model.load_state_dict(best_state)
    probs = predict_probs(model, valid_loader)
    exp_dir = CHECKPOINT_DIR / exp["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "input_dim": int(x_raw.shape[1]),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "feature_set": exp["feature_set"],
            "loss": exp["loss"],
            "best_epoch": int(best_epoch),
        },
        exp_dir / f"fold_{fold}.pt",
    )
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{exp['name']}_fold_{fold}_history.csv", index=False)
    return probs, {"fold": fold, "best_epoch": int(best_epoch), "best_mae": float(best_mae)}


def run_cv_experiment(exp, parts, y_0based, y_1based, splits):
    x_raw = matrix_for_feature_set(parts, exp["feature_set"])
    oof_probs = np.zeros((len(y_1based), 5), dtype=np.float32)
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        probs, fold_info = train_one_fold(exp, fold, train_idx, valid_idx, x_raw, y_0based, y_1based)
        oof_probs[valid_idx] = probs
        fold_info.update(
            {
                "experiment": exp["name"],
                "stage": exp["stage"],
                "feature_set": exp["feature_set"],
                "loss": exp["loss"],
                "feature_dim": int(x_raw.shape[1]),
            }
        )
        fold_rows.append(fold_info)
        print(f"{exp['name']} fold={fold} mae={fold_info['best_mae']:.4f} epoch={fold_info['best_epoch']}")
    return oof_probs, fold_rows, int(x_raw.shape[1])


def ceiling_estimates(y_1based):
    rng = np.random.default_rng(RUN_CONFIG["seed"])
    fake = y_1based.copy()
    perturb = rng.random(len(fake)) < 0.5
    for idx in np.where(perturb)[0]:
        if fake[idx] == 1:
            fake[idx] = 2
        elif fake[idx] == 5:
            fake[idx] = 4
        else:
            fake[idx] += rng.choice([-1, 1])
    random_pm1_mae = mean_absolute_error(y_1based, fake)

    majority = np.full_like(y_1based, 4)
    majority_metrics = {
        "mae": float(mean_absolute_error(y_1based, majority)),
        "accuracy": float(accuracy_score(y_1based, majority)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_1based, majority, weights="quadratic")),
    }

    values, counts = np.unique(y_1based, return_counts=True)
    probs = counts / counts.sum()
    prior_maes = []
    for _ in range(RUN_CONFIG["prior_sampling_repeats"]):
        sampled = rng.choice(values, size=len(y_1based), p=probs)
        prior_maes.append(mean_absolute_error(y_1based, sampled))

    rows = [
        {
            "baseline": "Random ±1 perturbation",
            "mae": float(random_pm1_mae),
            "accuracy": np.nan,
            "quadratic_weighted_kappa": np.nan,
            "note": "50% of labels moved to a valid adjacent class",
        },
        {
            "baseline": "Majority class = 4",
            "mae": majority_metrics["mae"],
            "accuracy": majority_metrics["accuracy"],
            "quadratic_weighted_kappa": majority_metrics["quadratic_weighted_kappa"],
            "note": "All samples predicted as confidence 4",
        },
        {
            "baseline": "Class prior random sampling",
            "mae": float(np.mean(prior_maes)),
            "accuracy": np.nan,
            "quadratic_weighted_kappa": np.nan,
            "note": f"Mean over {RUN_CONFIG['prior_sampling_repeats']} seeded draws; std={np.std(prior_maes):.4f}",
        },
        {
            "baseline": "Previous best: ensemble_avg_best_ordinal_towers",
            "mae": 0.5088009834289551,
            "accuracy": 0.5888097845929171,
            "quadratic_weighted_kappa": 0.3767550223211156,
            "note": "From confidence_model_true_5class_experiments.py",
        },
    ]
    return pd.DataFrame(rows)


def train_final_model(exp, parts, y_0based, train_epochs):
    seed_everything(RUN_CONFIG["seed"])
    x_raw = matrix_for_feature_set(parts, exp["feature_set"])
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)
    loader = DataLoader(ConfidenceDataset(x, y_0based), batch_size=RUN_CONFIG["batch_size"], shuffle=True)
    model = ConfidenceMLP(x_raw.shape[1], hidden=tuple(RUN_CONFIG["hidden"]), dropout=RUN_CONFIG["dropout"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RUN_CONFIG["learning_rate"], weight_decay=RUN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_epochs, 1))
    for _ in range(train_epochs):
        model.train()
        for batch in loader:
            x_batch = batch["x"].to(DEVICE)
            y_batch = batch["y"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(x_batch), y_batch, exp["loss"])
            loss.backward()
            optimizer.step()
        scheduler.step()
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x_raw.shape[1]),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "feature_set": exp["feature_set"],
            "loss": exp["loss"],
            "train_epochs": int(train_epochs),
        },
        CHECKPOINT_DIR / "best_final_full_bsd10k.pt",
    )
    return model, scaler


def predict_final(model, scaler, parts, feature_set):
    x_raw = matrix_for_feature_set(parts, feature_set)
    x = scaler.transform(x_raw).astype(np.float32)
    loader = DataLoader(ConfidenceDataset(x), batch_size=RUN_CONFIG["batch_size"], shuffle=False)
    return predict_probs(model, loader)


def markdown_table(df, columns=None, float_digits=4):
    view = df.copy() if columns is None else df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    view = view.fillna("")
    headers = [str(col) for col in view.columns]
    rows = [[str(value) for value in row] for row in view.to_numpy()]
    widths = [
        max([len(headers[idx])] + [len(row[idx]) for row in rows])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body)


def markdown_matrix(df, float_digits=4):
    view = df.copy()
    view.insert(0, "", view.index.astype(str))
    return markdown_table(view.reset_index(drop=True), float_digits=float_digits)


def write_ceiling_report(ceiling_df):
    text = [
        "# BSD10k Confidence Ceiling Estimate",
        "",
        markdown_table(ceiling_df, float_digits=4),
        "",
        "Random ±1 perturbation is a label-noise proxy, not an achievable model target. "
        "It estimates the error produced when half of labels are shifted to an adjacent valid ordinal class.",
        "",
    ]
    (ROOT / "ceiling_estimate.md").write_text("\n".join(text), encoding="utf-8")
    (REPORT_DIR / "ceiling_estimate.md").write_text("\n".join(text), encoding="utf-8")


def write_report(ceiling_df, summary_df, fold_df, best_exp, best_probs, y_1based, bsd35k_out, elapsed_seconds):
    stage1 = summary_df[summary_df["stage"] == "stage1"].sort_values("mae")
    stage2 = summary_df[summary_df["stage"] == "stage2"].sort_values("mae")
    previous = pd.DataFrame(
        [
            {
                "model": "previous best ensemble_avg_best_ordinal_towers",
                "mae": 0.5088009834289551,
                "spearman": np.nan,
                "accuracy": 0.5888097845929171,
                "quadratic_weighted_kappa": 0.3767550223211156,
                "macro_f1": 0.3338691751327909,
            },
            {
                "model": f"new best {best_exp['name']}",
                **metrics_from_probs(y_1based, best_probs),
            },
        ]
    )
    cm = pd.DataFrame(
        confusion_matrix(y_1based, best_probs.argmax(axis=1) + 1, labels=[1, 2, 3, 4, 5]),
        index=[f"true {i}" for i in range(1, 6)],
        columns=[f"pred {i}" for i in range(1, 6)],
    )
    pred_dist = (
        bsd35k_out["predicted_confidence_class"]
        .value_counts()
        .sort_index()
        .rename_axis("predicted_confidence_class")
        .reset_index(name="n")
    )
    pred_dist["rate"] = pred_dist["n"] / pred_dist["n"].sum()
    score_summary = bsd35k_out["predicted_confidence_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    best_metrics = metrics_from_probs(y_1based, best_probs)
    ceiling_mae = float(ceiling_df.loc[ceiling_df["baseline"] == "Random ±1 perturbation", "mae"].iloc[0])
    stage1_best = stage1.iloc[0]
    stage1_base = stage1[stage1["feature_set"] == "baseline"].iloc[0]
    stage2_ce = stage2[stage2["loss"] == "ce"].iloc[0]
    feature_gain = float(stage1_base["mae"] - stage1_best["mae"])
    loss_gain = float(stage2_ce["mae"] - stage2.iloc[0]["mae"])

    lines = [
        "# BSD10k Confidence 분류기 v2 재실험 보고서",
        "",
        f"작성일: 2026-05-10",
        f"실행 장치: {DEVICE}",
        f"총 실행 시간: {elapsed_seconds / 60:.1f}분",
        "",
        "## 1. Stage 0 ceiling estimate 결과",
        "",
        markdown_table(ceiling_df, float_digits=4),
        "",
        "## 2. Stage 1 feature ablation",
        "",
        markdown_table(
            stage1,
            ["experiment", "feature_set", "loss", "feature_dim", "mae", "spearman", "accuracy", "quadratic_weighted_kappa", "macro_f1"],
            float_digits=4,
        ),
        "",
        "## 3. Stage 2 loss 비교",
        "",
        markdown_table(
            stage2,
            ["experiment", "feature_set", "loss", "feature_dim", "mae", "spearman", "accuracy", "quadratic_weighted_kappa", "macro_f1"],
            float_digits=4,
        ),
        "",
        "## 4. 이전 best vs 새 best",
        "",
        markdown_table(previous, ["model", "mae", "spearman", "accuracy", "quadratic_weighted_kappa", "macro_f1"], float_digits=4),
        "",
        "## 5. Confusion matrix",
        "",
        markdown_matrix(cm, float_digits=0),
        "",
        "## 6. BSD35k-CS 적용 결과 요약",
        "",
        f"- 저장 파일: `outputs/confidence_model_v2/predictions/BSD35k-CS_predicted_v2.csv`",
        f"- 적용 row 수: {len(bsd35k_out):,}",
        f"- 평균 predicted confidence score: {bsd35k_out['predicted_confidence_score'].mean():.4f}",
        f"- 표준편차: {bsd35k_out['predicted_confidence_score'].std():.4f}",
        "",
        "예측 class 분포:",
        "",
        markdown_table(pred_dist, float_digits=4),
        "",
        "예측 score 요약:",
        "",
        markdown_matrix(score_summary.to_frame("predicted_confidence_score")),
        "",
        "## 7. 결론",
        "",
        f"1. 새 모델의 MAE {best_metrics['mae']:.4f}는 Random ±1 perturbation ceiling proxy {ceiling_mae:.4f}와 "
        f"{best_metrics['mae'] - ceiling_mae:+.4f} 차이다. 이 proxy가 human-level adjacent-label noise를 뜻한다면, "
        "모델은 그 근처까지 왔는지 여부를 이 차이로 판단할 수 있다.",
        "",
        f"2. feature engineering 개선폭은 baseline 대비 {feature_gain:+.4f} MAE 감소이고, "
        f"loss 변경 개선폭은 CE 대비 {loss_gain:+.4f} MAE 감소다. 이번 실행에서는 "
        f"{'feature engineering' if feature_gain >= loss_gain else 'ordinal-aware loss'} 쪽 기여가 더 컸다.",
        "",
        f"3. best Spearman은 {best_metrics['spearman']:.4f}이다. MAE가 ceiling proxy의 ±0.02 안이면 추가 모델 복잡도보다 "
        "라벨 품질/추가 annotation 점검이 우선이고, 그보다 멀면 confidence-relevant feature 추가가 다음 단계다.",
        "",
    ]
    report = "\n".join(lines)
    (ROOT / "confidence_model_v2_report_ko.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "confidence_model_v2_report_ko.md").write_text(report, encoding="utf-8")
    cm.to_csv(REPORT_DIR / "best_confusion_matrix.csv")


def select_best_experiment(summary_df):
    ordered = summary_df.sort_values(["mae", "spearman"], ascending=[True, False]).reset_index(drop=True)
    for _, candidate in ordered.iterrows():
        if not (ordered["spearman"] > float(candidate["spearman"]) + 0.02).any():
            return candidate
    return ordered.iloc[0]


def main():
    start = time.time()
    seed_everything(RUN_CONFIG["seed"])
    print("device:", DEVICE)

    parts, class_categories, top_class_categories = load_bsd10k_parts()
    y_1based = parts["df"]["confidence"].to_numpy(dtype=np.int64)
    y_0based = y_1based - 1
    print("BSD10k rows:", len(y_1based), dict(pd.Series(y_1based).value_counts().sort_index()))

    ceiling_df = ceiling_estimates(y_1based)
    ceiling_df.to_csv(REPORT_DIR / "ceiling_estimate.csv", index=False)
    write_ceiling_report(ceiling_df)

    splitter = StratifiedKFold(n_splits=RUN_CONFIG["folds"], shuffle=True, random_state=RUN_CONFIG["seed"])
    splits = list(splitter.split(np.zeros(len(y_1based)), y_1based))

    summary_rows = []
    fold_rows = []
    oof_records = {}

    for exp in STAGE1_EXPERIMENTS:
        probs, rows, feature_dim = run_cv_experiment(exp, parts, y_0based, y_1based, splits)
        row = {
            "experiment": exp["name"],
            "stage": exp["stage"],
            "feature_set": exp["feature_set"],
            "loss": exp["loss"],
            "feature_dim": feature_dim,
            **metrics_from_probs(y_1based, probs),
        }
        summary_rows.append(row)
        fold_rows.extend(rows)
        oof_records[exp["name"]] = {"exp": exp, "probs": probs, "folds": rows}

    stage1_df = pd.DataFrame(summary_rows)
    stage1_best_feature = stage1_df.sort_values(["mae", "spearman"], ascending=[True, False]).iloc[0]["feature_set"]
    print("Stage 1 best feature set:", stage1_best_feature)

    for loss in STAGE2_LOSSES:
        exp = {
            "name": f"stage2_{stage1_best_feature}_{loss}",
            "stage": "stage2",
            "feature_set": stage1_best_feature,
            "loss": loss,
        }
        if loss == "ce":
            matching = [name for name, item in oof_records.items() if item["exp"]["feature_set"] == stage1_best_feature and item["exp"]["loss"] == "ce"]
            if matching:
                probs = oof_records[matching[0]]["probs"]
                feature_dim = int(matrix_for_feature_set(parts, stage1_best_feature).shape[1])
                row = {
                    "experiment": exp["name"],
                    "stage": exp["stage"],
                    "feature_set": exp["feature_set"],
                    "loss": exp["loss"],
                    "feature_dim": feature_dim,
                    **metrics_from_probs(y_1based, probs),
                }
                summary_rows.append(row)
                continue
        probs, rows, feature_dim = run_cv_experiment(exp, parts, y_0based, y_1based, splits)
        row = {
            "experiment": exp["name"],
            "stage": exp["stage"],
            "feature_set": exp["feature_set"],
            "loss": exp["loss"],
            "feature_dim": feature_dim,
            **metrics_from_probs(y_1based, probs),
        }
        summary_rows.append(row)
        fold_rows.extend(rows)
        oof_records[exp["name"]] = {"exp": exp, "probs": probs, "folds": rows}

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_df = summary_df.sort_values(["mae", "spearman"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(REPORT_DIR / "v2_experiment_summary.csv", index=False)
    fold_df.to_csv(REPORT_DIR / "v2_fold_metrics.csv", index=False)

    selected = select_best_experiment(summary_df)
    best_name = selected["experiment"]
    best_record = oof_records.get(best_name)
    if best_record is None:
        source_name = [
            name
            for name, item in oof_records.items()
            if item["exp"]["feature_set"] == selected["feature_set"] and item["exp"]["loss"] == selected["loss"]
        ][0]
        best_record = oof_records[source_name]
    best_exp = {
        "name": best_name,
        "stage": str(selected["stage"]),
        "feature_set": str(selected["feature_set"]),
        "loss": str(selected["loss"]),
    }
    best_probs = best_record["probs"]

    oof_out = parts["df"].copy()
    oof_out["best_experiment"] = best_name
    oof_out["true_confidence"] = y_1based
    oof_out["predicted_confidence_class"] = best_probs.argmax(axis=1) + 1
    oof_out["predicted_confidence_score"] = expected_score_from_probs(best_probs)
    for idx in range(5):
        oof_out[f"prob_confidence_{idx + 1}"] = best_probs[:, idx]
    oof_out.to_csv(PRED_DIR / "BSD10k_oof_predicted_v2.csv", index=False)

    best_fold_epochs = [row["best_epoch"] + 1 for row in best_record.get("folds", [])]
    train_epochs = int(round(float(np.median(best_fold_epochs)))) if best_fold_epochs else RUN_CONFIG["epochs"]
    train_epochs = max(1, min(RUN_CONFIG["epochs"], train_epochs))
    print("best:", best_name, "final train epochs:", train_epochs)
    final_model, final_scaler = train_final_model(best_exp, parts, y_0based, train_epochs)

    bsd35k_parts = load_bsd35k_parts(class_categories, top_class_categories, parts)
    bsd35k_probs = predict_final(final_model, final_scaler, bsd35k_parts, best_exp["feature_set"])
    bsd35k_out = bsd35k_parts["df"].copy()
    bsd35k_out["best_experiment"] = best_name
    bsd35k_out["predicted_confidence_class"] = bsd35k_probs.argmax(axis=1) + 1
    bsd35k_out["predicted_confidence_score"] = expected_score_from_probs(bsd35k_probs)
    for idx in range(5):
        bsd35k_out[f"prob_confidence_{idx + 1}"] = bsd35k_probs[:, idx]
    bsd35k_out.to_csv(PRED_DIR / "BSD35k-CS_predicted_v2.csv", index=False)

    bsd35k_summary = {
        "best_experiment": best_name,
        "best_feature_set": best_exp["feature_set"],
        "best_loss": best_exp["loss"],
        "final_train_epochs": train_epochs,
        "bsd35k_rows": int(len(bsd35k_out)),
        "mean_predicted_confidence_score": float(bsd35k_out["predicted_confidence_score"].mean()),
        "predicted_class_distribution": {
            str(k): int(v)
            for k, v in bsd35k_out["predicted_confidence_class"].value_counts().sort_index().items()
        },
        "prediction_file": "outputs/confidence_model_v2/predictions/BSD35k-CS_predicted_v2.csv",
        "elapsed_seconds": time.time() - start,
    }
    with open(REPORT_DIR / "v2_run_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_config": RUN_CONFIG,
                "best": selected.to_dict(),
                "bsd35k": bsd35k_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_report(ceiling_df, summary_df, fold_df, best_exp, best_probs, y_1based, bsd35k_out, time.time() - start)
    print(summary_df)
    print(json.dumps(bsd35k_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
