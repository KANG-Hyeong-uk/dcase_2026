from __future__ import annotations

import json
import os
import pickle
import random
import runpy
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader


SEED = 1821
THRESHOLDS = (2, 3, 4)


def find_repo_root(start: str | Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "dcase2026_task1_baseline").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find repo root containing dcase2026_task1_baseline and data.")


ROOT = find_repo_root()
BASELINE_DIR = ROOT / "dcase2026_task1_baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from dataset_utils import HATRDataset  # noqa: E402
from evaluate import evaluate_model  # noqa: E402
from losses import CrossEntropyLoss  # noqa: E402
from models import BaseClassifier  # noqa: E402
from train_test import make_serializable, train_model  # noqa: E402
from utils import build_class_to_topclass_mapping, set_seed  # noqa: E402


def seed_everything(seed: int = SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    return seed


def ensure_processed_dataset(root: Path = ROOT) -> None:
    processed = root / "data" / "processed_dataset.csv"
    if processed.exists():
        return
    cwd = Path.cwd()
    os.chdir(root)
    try:
        runpy.run_path(str(root / "dcase2026_task1_baseline" / "build_dataset.py"), run_name="__main__")
    finally:
        os.chdir(cwd)


def load_baseline_assets(root: Path = ROOT):
    ensure_processed_dataset(root)
    processed = pd.read_csv(root / "data" / "processed_dataset.csv")
    processed["index"] = processed["index"].astype(str)
    with open(root / "data" / "class_dict.json", "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    with open(root / "data" / "top_class_dict.json", "r", encoding="utf-8") as f:
        top_class_dict = json.load(f)
    return processed, class_dict, top_class_dict


def make_fixed_holdout(
    df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = SEED,
    test_size: float = 0.2,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "fixed_train_pool_final_test_split.csv"

    if split_path.exists():
        split_df = pd.read_csv(split_path, dtype={"sound_id": str})
        train_ids = set(split_df.loc[split_df["split"] == "train_pool", "sound_id"])
        test_ids = set(split_df.loc[split_df["split"] == "final_test", "sound_id"])
        train_pool = df[df["index"].isin(train_ids)].reset_index(drop=True)
        final_test = df[df["index"].isin(test_ids)].reset_index(drop=True)
        return train_pool, final_test, split_df

    labels = df["class_idx"].to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_pool = df.iloc[train_idx].reset_index(drop=True)
    final_test = df.iloc[test_idx].reset_index(drop=True)

    split_df = pd.concat(
        [
            train_pool[["index", "class", "class_idx"]].assign(split="train_pool"),
            final_test[["index", "class", "class_idx"]].assign(split="final_test"),
        ],
        ignore_index=True,
    ).rename(columns={"index": "sound_id"})
    split_df.to_csv(split_path, index=False)
    return train_pool, final_test, split_df


def load_prediction_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "sound_id" not in df.columns:
        raise ValueError(f"{path} has no sound_id column.")
    df["sound_id"] = df["sound_id"].astype(str)
    return df


def merge_scores_to_train_pool(
    train_pool: pd.DataFrame,
    score_df: pd.DataFrame,
    score_col: str = "predicted_confidence_score",
    missing_score_policy: str = "error",
) -> pd.DataFrame:
    if score_col not in score_df.columns:
        raise ValueError(f"score_df has no {score_col!r} column.")
    keep_cols = ["sound_id", score_col] + [
        c for c in score_df.columns if c not in {"sound_id", score_col} and c.startswith(("prob_", "rank_", "binary_", "fiveclass_"))
    ]
    merged = train_pool.merge(score_df[keep_cols], left_on="index", right_on="sound_id", how="left")
    missing = merged[score_col].isna().sum()
    if missing:
        examples = merged.loc[merged[score_col].isna(), "index"].head(5).tolist()
        if missing_score_policy == "drop":
            print(
                f"[merge_scores_to_train_pool] dropping {missing} train_pool rows with missing confidence scores. "
                f"Examples: {examples}"
            )
            merged = merged[merged[score_col].notna()].reset_index(drop=True)
        else:
            raise ValueError(f"Missing confidence scores for {missing} train_pool rows. Examples: {examples}")
    return merged.drop(columns=["sound_id"])


def confidence_threshold_specs(thresholds=THRESHOLDS):
    return [{"label": f"pred_ge_{t}", "threshold": float(t), "display_threshold": int(t)} for t in thresholds]


def write_filter_counts(
    scored_train_pool: pd.DataFrame,
    output_root: str | Path,
    score_col: str = "predicted_confidence_score",
    threshold_specs=None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    threshold_specs = confidence_threshold_specs() if threshold_specs is None else threshold_specs
    rows = []
    total = len(scored_train_pool)
    for spec in threshold_specs:
        label = spec["label"]
        threshold = float(spec["threshold"])
        kept = scored_train_pool[scored_train_pool[score_col] >= threshold]
        row = {
            "filter_label": label,
            "score_column": score_col,
            "threshold": threshold,
            "display_threshold": spec.get("display_threshold", threshold),
            "train_pool_samples": int(total),
            "retained_samples": int(len(kept)),
            "dropped_samples": int(total - len(kept)),
            "retained_ratio": float(len(kept) / total) if total else 0.0,
            "num_classes_retained": int(kept["class"].nunique()) if "class" in kept.columns else None,
        }
        rows.append(row)
    counts = pd.DataFrame(rows)
    counts.to_csv(output_root / "filter_counts.csv", index=False)
    print("\n=== Filter counts ===")
    print(counts.to_string(index=False))
    return counts


def _can_stratify(y: pd.Series, valid_size: float) -> bool:
    counts = y.value_counts()
    if counts.min() < 2:
        return False
    n_valid = int(np.ceil(len(y) * valid_size))
    return n_valid >= y.nunique()


def make_train_val_split(df: pd.DataFrame, seed: int, valid_size: float):
    stratify = df["class_idx"] if _can_stratify(df["class_idx"], valid_size) else None
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=valid_size,
        random_state=seed,
        stratify=stratify,
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def build_loader(df: pd.DataFrame, batch_size: int, shuffle: bool, aug: bool, mask_pct: float = 0.7):
    dataset = HATRDataset(df.reset_index(drop=True), aug=aug, mask_pct=mask_pct)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode="fan_out")
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)


def write_confusion_matrix(output_dir: str | Path, class_dict: dict, title: str):
    output_dir = Path(output_dir)
    pred_path = output_dir / "evaluation" / "predictions.csv"
    pred_df = pd.read_csv(pred_path)
    class_labels = [name for name, _ in sorted(class_dict.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(
        pred_df["ground_truth"],
        pred_df["prediction"],
        labels=class_labels,
        normalize="true",
    )
    cm = np.nan_to_num(cm)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(output_dir / "evaluation" / "confusion_matrix_normalized_true.csv")

    fig_size = max(10, len(class_labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)
    fig.tight_layout()
    annotate_confusion_matrix(ax, cm, fontsize=7)
    fig.savefig(output_dir / "evaluation" / "confusion_matrix_normalized_true.png", dpi=180)
    plt.close(fig)
    return cm_df


def device_summary() -> str:
    if not torch.cuda.is_available():
        return "cpu (CUDA unavailable)"
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"cuda:0 ({name}, {total_gb:.1f} GB)"


def read_metrics_file(path: str | Path) -> dict:
    path = Path(path)
    metrics = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().replace("%", "")
        try:
            metrics[key.strip()] = float(value)
        except ValueError:
            continue
    return metrics


def train_and_evaluate_one(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    final_test_df: pd.DataFrame,
    class_dict: dict,
    top_class_dict: dict,
    output_dir: str | Path,
    mode: str = "both",
    seed: int = SEED,
    batch_size: int = 64,
    num_epochs: int = 100,
    lr: float = 0.001,
    patience: int = 5,
    early_stopping_factor: int = 3,
):
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_size_audio = 512 if mode in ["audio", "both"] else 0
    emb_size_text = 512 if mode in ["text", "both"] else 0

    model = BaseClassifier(
        hidden_size=128,
        num_classes=len(class_dict),
        emb_size_audio=emb_size_audio,
        emb_size_text=emb_size_text,
        dropout=0.1,
        use_batch_norm=True,
        mode=mode,
    ).to(device)
    model.apply(init_weights)

    train_loader = build_loader(train_df, batch_size=batch_size, shuffle=True, aug=True)
    val_loader = build_loader(val_df, batch_size=batch_size, shuffle=False, aug=False)
    test_loader = build_loader(final_test_df, batch_size=batch_size, shuffle=False, aug=False)

    print(
        f"\n[run] {output_dir}\n"
        f"  device={device_summary()}\n"
        f"  samples: train={len(train_df)}, val={len(val_df)}, final_test={len(final_test_df)}\n"
        f"  batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}\n"
        f"  max_epochs={num_epochs}, batch_size={batch_size}"
    )

    started = time.time()
    best_accuracy, history, trained_model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        lr=lr,
        classification_weight=1.0,
        classification_criterion=CrossEntropyLoss(),
        output_dir=str(output_dir),
        scheduler_type="step",
        patience=patience,
        early_stopping_factor=early_stopping_factor,
    )
    elapsed_min = (time.time() - started) / 60.0

    history["model_info"] = {
        "model_class": trained_model.__class__.__name__,
        "mode": mode,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "random_seed": seed,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "final_test_samples": int(len(final_test_df)),
        "elapsed_train_minutes": float(elapsed_min),
        "device": device_summary(),
    }
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(make_serializable(history), f, indent=2, ensure_ascii=False)

    metrics = evaluate_model(
        model_class=BaseClassifier,
        model_path=str(output_dir / "best_model.pth"),
        data_loader=test_loader,
        device=device,
        class_to_topclass=build_class_to_topclass_mapping(class_dict, top_class_dict),
        output_dir=str(output_dir),
        fold_id="final_test",
        class_dict=class_dict,
    )
    write_confusion_matrix(output_dir, class_dict, title=f"{output_dir.name} normalized confusion matrix")
    print(f"  completed in {elapsed_min:.1f} min")
    return best_accuracy, metrics


def run_downstream_grid(
    *,
    experiment_name: str,
    score_df: pd.DataFrame,
    output_root: str | Path,
    score_col: str = "predicted_confidence_score",
    threshold_specs=None,
    baseline_modes=("both",),
    use_kfold: bool = True,
    n_folds: int = 5,
    seed: int = SEED,
    missing_score_policy: str = "error",
    valid_size: float = 0.2,
    batch_size: int = 64,
    num_epochs: int = 100,
    lr: float = 0.001,
    patience: int = 5,
    early_stopping_factor: int = 3,
):
    seed_everything(seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_specs = confidence_threshold_specs() if threshold_specs is None else threshold_specs

    full_df, class_dict, top_class_dict = load_baseline_assets(ROOT)
    train_pool, final_test, split_df = make_fixed_holdout(full_df, output_root, seed=seed)
    scored_train_pool = merge_scores_to_train_pool(
        train_pool,
        score_df,
        score_col=score_col,
        missing_score_policy=missing_score_policy,
    )

    scored_train_pool.to_csv(output_root / "train_pool_with_confidence_scores.csv", index=False)
    final_test.to_csv(output_root / "final_test_holdout_unfiltered.csv", index=False)
    filter_counts = write_filter_counts(
        scored_train_pool,
        output_root,
        score_col=score_col,
        threshold_specs=threshold_specs,
    )

    planned_runs = len(threshold_specs) * len(baseline_modes) * (n_folds if use_kfold else 1)
    print(
        "\n=== Downstream baseline plan ===\n"
        f"experiment={experiment_name}\n"
        f"device={device_summary()}\n"
        f"baseline_modes={baseline_modes}\n"
        f"use_kfold={use_kfold}, n_folds={n_folds}\n"
        f"thresholds={[spec['display_threshold'] for spec in threshold_specs]}\n"
        f"planned_trainings={planned_runs}\n"
        f"num_epochs={num_epochs}, batch_size={batch_size}"
    )

    rows = []
    for spec in threshold_specs:
        label = spec["label"]
        threshold = float(spec["threshold"])
        display_threshold = spec.get("display_threshold", threshold)
        filtered = scored_train_pool[scored_train_pool[score_col] >= threshold].reset_index(drop=True)
        filtered_dir = output_root / label
        filtered_dir.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(filtered_dir / "filtered_train_pool.csv", index=False)

        base_row = {
            "experiment": experiment_name,
            "filter_label": label,
            "score_column": score_col,
            "threshold": threshold,
            "display_threshold": display_threshold,
            "train_pool_samples": int(len(scored_train_pool)),
            "retained_samples": int(len(filtered)),
            "retained_ratio": float(len(filtered) / len(scored_train_pool)) if len(scored_train_pool) else 0.0,
            "final_test_samples": int(len(final_test)),
        }
        if len(filtered) < 10 or filtered["class_idx"].nunique() < 2:
            rows.append({**base_row, "status": "skipped_too_few_samples"})
            continue

        split_iter = []
        if use_kfold:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(filtered)), filtered["class_idx"])):
                split_iter.append((f"fold_{fold}", filtered.iloc[train_idx].reset_index(drop=True), filtered.iloc[val_idx].reset_index(drop=True)))
        else:
            tr_df, va_df = make_train_val_split(filtered, seed=seed, valid_size=valid_size)
            split_iter.append(("single_split", tr_df, va_df))

        for mode in baseline_modes:
            for split_name, tr_df, va_df in split_iter:
                run_dir = filtered_dir / mode / split_name
                existing_metrics_path = run_dir / "evaluation" / "results.txt"
                if existing_metrics_path.exists():
                    metrics = read_metrics_file(existing_metrics_path)
                    rows.append(
                        {
                            **base_row,
                            "status": "completed_existing",
                            "mode": mode,
                            "split": split_name,
                            "train_samples": int(len(tr_df)),
                            "val_samples": int(len(va_df)),
                            **metrics,
                            "output_dir": str(run_dir),
                        }
                    )
                    print(f"[skip] existing completed run: {run_dir}")
                    pd.DataFrame(rows).to_csv(output_root / "summary_results.csv", index=False)
                    continue

                best_val_acc, metrics = train_and_evaluate_one(
                    tr_df,
                    va_df,
                    final_test,
                    class_dict,
                    top_class_dict,
                    run_dir,
                    mode=mode,
                    seed=seed,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    lr=lr,
                    patience=patience,
                    early_stopping_factor=early_stopping_factor,
                )
                rows.append(
                    {
                        **base_row,
                        "status": "ok",
                        "mode": mode,
                        "split": split_name,
                        "train_samples": int(len(tr_df)),
                        "val_samples": int(len(va_df)),
                        "best_val_accuracy": float(best_val_acc),
                        **metrics,
                        "output_dir": str(run_dir),
                    }
                )
                pd.DataFrame(rows).to_csv(output_root / "summary_results.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary_results.csv", index=False)
    return summary, filter_counts


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_outputs: int = 5, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, num_outputs))

    def forward(self, x):
        return self.net(x)


class ConfidenceRegressor(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, 1))

    def forward(self, x):
        return self.net(x)


def _load_metadata_for_confidence(root: Path = ROOT) -> pd.DataFrame:
    meta = pd.read_csv(root / "data" / "metadata" / "BSD10k_metadata.csv")
    meta["sound_id"] = meta["sound_id"].astype(str)
    meta["class_idx_raw"] = meta["class_idx"].astype(str)
    return meta


def _class_mapping_from_metadata(root: Path, expected_dim: int | None = None):
    meta = _load_metadata_for_confidence(root)
    classes = sorted(meta["class_idx_raw"].dropna().astype(str).unique())
    if expected_dim is not None and len(classes) != expected_dim:
        raise ValueError(f"Expected {expected_dim} class one-hot columns, got {len(classes)} from metadata.")
    return {label: idx for idx, label in enumerate(classes)}


def _build_confidence_features(
    ids: pd.Series,
    root: Path = ROOT,
    class_mapping: dict[str, int] | None = None,
    expected_class_dim: int | None = None,
    return_skipped: bool = False,
    progress_label: str | None = None,
):
    ids = ids.astype(str).tolist()
    meta = _load_metadata_for_confidence(root).set_index("sound_id")
    if class_mapping is None:
        class_mapping = _class_mapping_from_metadata(root, expected_dim=expected_class_dim)
    class_dim = len(class_mapping)

    audio_dir = root / "data" / "features" / "clap_audio_embeddings"
    text_dir = root / "data" / "features" / "clap_text_embeddings"
    rows = []
    kept_ids = []
    skipped = []
    total = len(ids)
    for n, sid in enumerate(ids, start=1):
        if progress_label and (n == 1 or n % 1000 == 0 or n == total):
            print(f"[{progress_label}] loading features {n:,}/{total:,} | kept={len(kept_ids):,} | skipped={len(skipped):,}")
        audio_path = audio_dir / f"{sid}.npy"
        text_path = text_dir / f"{sid}.npy"
        if sid not in meta.index or not audio_path.exists() or not text_path.exists():
            skipped.append((sid, "missing_metadata_or_embedding"))
            continue
        onehot = np.zeros(class_dim, dtype=np.float32)
        raw_class = str(meta.loc[sid, "class_idx_raw"])
        if raw_class not in class_mapping:
            skipped.append((sid, f"class_not_in_mapping:{raw_class}"))
            continue
        try:
            audio = np.load(audio_path).astype(np.float32).reshape(-1)
            text = np.load(text_path).astype(np.float32).reshape(-1)
        except Exception as exc:
            skipped.append((sid, f"embedding_load_failed:{type(exc).__name__}"))
            continue
        onehot[class_mapping[raw_class]] = 1.0
        rows.append(np.concatenate([audio, text, onehot]).astype(np.float32))
        kept_ids.append(sid)
    if not rows:
        raise ValueError("No confidence features could be built for the requested ids.")
    result = (kept_ids, np.stack(rows).astype(np.float32), class_mapping)
    if return_skipped:
        return (*result, skipped)
    return result


@torch.no_grad()
def load_517_mlp_classification_scores(train_pool: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    ckpt_path = root / "outputs" / "confidence_mlp_outputs" / "best_model_C_audio_text_class_dropout_0.5.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[517 MLP classification] inference device: {device}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = int(ckpt["config"]["input_dim"])
    dropout = float(ckpt["config"].get("selected_dropout", ckpt.get("dropout", 0.5)))
    expected_class_dim = input_dim - 1024
    ids, x, _, skipped = _build_confidence_features(
        train_pool["index"],
        root=root,
        expected_class_dim=expected_class_dim,
        return_skipped=True,
        progress_label="517 MLP classification",
    )
    model = SimpleMLP(input_dim=input_dim, num_outputs=5, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    probs = F.softmax(model(torch.tensor(x, dtype=torch.float32, device=device)), dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    pred_class = pred_idx.astype(np.float32) + 1.0
    out = pd.DataFrame({"sound_id": ids, "predicted_confidence_score": pred_class})
    for i in range(5):
        out[f"prob_confidence_{i + 1}"] = probs[:, i]
    out["predicted_confidence_class"] = pred_class.astype(int)
    if skipped:
        print(f"[517 MLP classification] skipped {len(skipped)} unreadable/missing feature rows.")
        print(f"[517 MLP classification] skipped examples: {skipped[:5]}")
    return out


@torch.no_grad()
def load_517_regression_scores(train_pool: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    root = Path(root)
    cache_path = root / "baseline_confidnce_train" / "outputs" / "517_regression" / "train_pool_regression_scores.csv"
    if cache_path.exists():
        print(f"[517 regression] loading cached scores: {cache_path}")
        return load_prediction_csv(cache_path)

    model_path = root / "outputs" / "confidence_regression_outputs" / "C_audio_text_class_dropout_0.5_model.pt"
    encoder_path = root / "outputs" / "confidence_regression_outputs" / "class_label_encoder.pkl"
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    classes = [str(c) for c in encoder.classes_]
    class_mapping = {label: idx for idx, label in enumerate(classes)}
    ids, x, _, skipped = _build_confidence_features(
        train_pool["index"],
        root=root,
        class_mapping=class_mapping,
        expected_class_dim=len(classes),
        return_skipped=True,
        progress_label="517 regression",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[517 regression] inference device: {device}")
    model = ConfidenceRegressor(input_dim=x.shape[1], dropout=0.5).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    score = model(torch.tensor(x, dtype=torch.float32, device=device)).view(-1).cpu().numpy()
    out = pd.DataFrame({"sound_id": ids, "predicted_confidence_score": score})
    if skipped:
        print(f"[517 regression] skipped {len(skipped)} unreadable/missing feature rows.")
        print(f"[517 regression] skipped examples: {skipped[:5]}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    print(f"[517 regression] saved scores: {cache_path}")
    return out


RESULT_COLUMNS = [
    "filter_label",
    "threshold",
    "retained_samples",
    "retained_ratio",
    "mode",
    "split",
    "train_samples",
    "val_samples",
    "final_test_samples",
    "best_val_accuracy",
    "accuracy",
    "top_accuracy",
    "macro_accuracy",
    "hierarchical_accuracy",
    "hierarchical_precision",
    "hierarchical_recall",
    "hierarchical_f1",
    "output_dir",
]

FOLD_SUMMARY_METRICS = [
    "accuracy",
    "hierarchical_accuracy",
    "hierarchical_f1",
    "hierarchical_precision",
    "hierarchical_recall",
    "macro_accuracy",
    "macro_top_accuracy",
    "top_accuracy",
]


def load_downstream_summary(output_root: str | Path) -> pd.DataFrame:
    output_root = Path(output_root)
    summary_path = output_root / "summary_results.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    numeric_cols = [
        "threshold",
        "retained_samples",
        "retained_ratio",
        "train_samples",
        "val_samples",
        "final_test_samples",
        "best_val_accuracy",
        "accuracy",
        "top_accuracy",
        "macro_accuracy",
        "macro_top_accuracy",
        "hierarchical_accuracy",
        "hierarchical_precision",
        "hierarchical_recall",
        "hierarchical_f1",
    ]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    if "split" in summary.columns:
        summary["fold_num"] = summary["split"].astype(str).str.extract(r"fold_(\d+)").astype(float)
    return summary


def print_all_downstream_results(
    output_root: str | Path,
    metric: str = "hierarchical_accuracy",
    sort: bool = True,
) -> pd.DataFrame:
    output_root = Path(output_root)
    summary = load_downstream_summary(output_root)
    if sort and metric in summary.columns:
        summary = summary.sort_values(metric, ascending=False)
    elif {"threshold", "fold_num"}.issubset(summary.columns):
        summary = summary.sort_values(["threshold", "fold_num"])

    cols = [col for col in RESULT_COLUMNS if col in summary.columns]
    view = summary[cols].copy()
    out_path = output_root / "all_run_results_view.csv"
    view.to_csv(out_path, index=False)

    print("\n=== All downstream run results ===")
    print(f"source: {output_root / 'summary_results.csv'}")
    print(f"saved view: {out_path}")
    print(view.to_string(index=False))
    return view


def summarize_folds_by_confidence(
    output_root: str | Path,
    metrics: list[str] | tuple[str, ...] = FOLD_SUMMARY_METRICS,
    group_cols=("filter_label", "mode"),
) -> pd.DataFrame:
    output_root = Path(output_root)
    summary = load_downstream_summary(output_root)
    group_cols = tuple(col for col in group_cols if col in summary.columns)
    metrics = [col for col in metrics if col in summary.columns]
    if not group_cols:
        raise ValueError("No grouping columns are available for fold summary.")
    if not metrics:
        raise ValueError("No metric columns are available for fold summary.")

    rows = []
    ok = summary[summary[metrics].notna().any(axis=1)].copy()
    for keys, group in ok.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in ["threshold", "display_threshold", "retained_samples", "retained_ratio", "final_test_samples"]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        row["fold_count"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)

    stats = pd.DataFrame(rows)
    sort_cols = [col for col in ["threshold", "filter_label", "mode"] if col in stats.columns]
    if sort_cols:
        stats = stats.sort_values(sort_cols).reset_index(drop=True)

    out_path = output_root / "fold_metric_summary_by_confidence.csv"
    stats.to_csv(out_path, index=False)

    print("\n=== Fold mean \u00b1 std by confidence threshold ===")
    print(f"source: {output_root / 'summary_results.csv'}")
    print(f"saved summary: {out_path}")
    for _, row in stats.iterrows():
        label = row.get("filter_label", "")
        display_threshold = row.get("display_threshold", row.get("threshold", ""))
        if pd.notna(display_threshold):
            threshold_name = f"confidence{int(display_threshold)}"
        elif str(label).startswith("pred_ge_"):
            threshold_name = f"confidence{str(label).replace('pred_ge_', '')}"
        else:
            threshold_name = str(label)
        print(f"\n{threshold_name}:")
        for metric in metrics:
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            if pd.notna(mean):
                print(f"  {metric:<22}: {mean:.2f}% \u00b1 {std:.2f}%")
    return stats


def select_best_folds(
    output_root: str | Path,
    metric: str = "hierarchical_accuracy",
    group_cols=("filter_label", "mode"),
) -> pd.DataFrame:
    summary = load_downstream_summary(output_root)
    if metric not in summary.columns:
        raise ValueError(f"Metric {metric!r} is not in summary columns.")
    ok = summary[summary[metric].notna()].copy()
    if ok.empty:
        raise ValueError(f"No rows have a numeric {metric!r}.")
    idx = ok.groupby(list(group_cols))[metric].idxmax()
    best = ok.loc[idx].sort_values(list(group_cols)).reset_index(drop=True)
    best.to_csv(Path(output_root) / f"best_folds_by_{metric}.csv", index=False)
    return best


def _read_confusion_matrix_csv(run_dir: str | Path) -> pd.DataFrame:
    cm_path = Path(run_dir) / "evaluation" / "confusion_matrix_normalized_true.csv"
    if not cm_path.exists():
        raise FileNotFoundError(cm_path)
    return pd.read_csv(cm_path, index_col=0)


def annotate_confusion_matrix(ax, cm: np.ndarray, fontsize: int = 7) -> None:
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = float(cm[i, j])
            text_color = "white" if value >= 0.45 else "black"
            stroke_color = "black" if text_color == "white" else "white"
            text = ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=1.25, foreground=stroke_color)]
            )


def _plot_one_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str,
    save_path: str | Path,
    figsize=(18, 15),
    annot_fontsize: int = 7,
    show: bool = False,
) -> Path:
    labels = cm_df.index.astype(str).tolist()
    cm = cm_df.to_numpy(dtype=float)
    save_path = Path(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized true-class ratio", rotation=270, labelpad=18)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    annotate_confusion_matrix(ax, cm, fontsize=annot_fontsize)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_best_fold_confusion_matrices(
    output_root: str | Path,
    metric: str = "hierarchical_accuracy",
    group_cols=("filter_label", "mode"),
    save_name: str | None = None,
    show: bool = True,
) -> tuple[pd.DataFrame, Path]:
    output_root = Path(output_root)
    best = select_best_folds(output_root, metric=metric, group_cols=group_cols)
    n = len(best)
    if n == 0:
        raise ValueError("No best folds to plot.")

    fig_height = max(10, 9.0 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(18, fig_height), squeeze=False)
    individual_paths = []

    for ax, (_, row) in zip(axes[:, 0], best.iterrows()):
        cm_df = _read_confusion_matrix_csv(row["output_dir"])
        labels = cm_df.index.astype(str).tolist()
        cm = cm_df.to_numpy(dtype=float)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        threshold = row.get("threshold", "")
        fold = row.get("split", "")
        score = row.get(metric, np.nan)
        retained = row.get("retained_samples", np.nan)
        ratio = row.get("retained_ratio", np.nan)
        ax.set_title(
            f"{row.get('filter_label')} | {fold} | best {metric}={score:.2f} | "
            f"retained={int(retained):,} ({ratio:.1%}) | threshold={threshold}",
            fontsize=11,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized true-class ratio", rotation=270, labelpad=15)
        annotate_confusion_matrix(ax, cm, fontsize=6)

        one_path = output_root / f"best_fold_confusion_matrix_{row.get('filter_label')}_{row.get('mode')}.png"
        individual_paths.append(
            _plot_one_confusion_matrix(
                cm_df,
                title=(
                    f"{row.get('filter_label')} | {row.get('split')} | "
                    f"best {metric}={score:.2f} | retained={int(retained):,} ({ratio:.1%})"
                ),
                save_path=one_path,
                figsize=(18, 15),
                annot_fontsize=8,
                show=False,
            )
        )

    fig.tight_layout()
    if save_name is None:
        save_name = f"best_fold_confusion_matrices_by_{metric}.png"
    save_path = output_root / save_name
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    best_path = output_root / f"best_folds_by_{metric}.csv"
    print("\n=== Best folds selected for confusion matrix ===")
    show_cols = [
        "filter_label",
        "threshold",
        "mode",
        "split",
        "retained_samples",
        "retained_ratio",
        "best_val_accuracy",
        "accuracy",
        "hierarchical_f1",
        "output_dir",
    ]
    show_cols = [col for col in show_cols if col in best.columns]
    print(best[show_cols].to_string(index=False))
    print(f"\nsaved best-fold table: {best_path}")
    print(f"saved confusion matrix figure: {save_path}")
    print("saved individual confusion matrix figures:")
    for path in individual_paths:
        print(f"  {path}")
    return best, save_path


def show_v2_5class_results(
    output_root: str | Path | None = None,
    metric: str = "hierarchical_accuracy",
    sort: bool = True,
    show_plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    if output_root is None:
        output_root = ROOT / "baseline_confidnce_train" / "outputs" / "v2_5class"
    output_root = Path(output_root)
    all_results = print_all_downstream_results(output_root, metric=metric, sort=sort)
    best, fig_path = plot_best_fold_confusion_matrices(
        output_root,
        metric=metric,
        group_cols=("filter_label", "mode"),
        show=show_plot,
    )
    summarize_folds_by_confidence(output_root)
    return all_results, best, fig_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect downstream baseline confidence-filter results.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "baseline_confidnce_train" / "outputs" / "v2_5class"),
        help="Experiment output directory containing summary_results.csv.",
    )
    parser.add_argument("--metric", default="hierarchical_accuracy", help="Metric used to sort rows and pick the best fold.")
    parser.add_argument("--no-sort", action="store_true", help="Keep threshold/fold order instead of sorting by metric descending.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening an interactive window.")
    args = parser.parse_args()

    show_v2_5class_results(
        output_root=args.output_root,
        metric=args.metric,
        sort=not args.no_sort,
        show_plot=not args.no_show,
    )
