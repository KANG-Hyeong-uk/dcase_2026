import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models import BaseClassifier


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASELINE_DIR / "config.yaml"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_target_classes(df):
    df = df.copy()
    df["sound_id"] = df["sound_id"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    df["class_top"] = df["class_top"].fillna(df["class"].str.split("-").str[0])
    s = df["class_idx"].astype(str)
    keep = ~((s.str.len() == 3) & (s.str.endswith("99") | s.str.endswith("00")))
    return df[keep].copy()


def count_existing_embeddings(df, folder):
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum((folder / f"{sound_id}.npy").is_file() for sound_id in df["sound_id"].astype(str))


def pick_embedding_folder(df, configured_folder, candidates, label):
    configured_folder = resolve_path(configured_folder)
    choices = [configured_folder] + [ROOT / c for c in candidates]
    scored = [(count_existing_embeddings(df, folder), folder) for folder in choices]
    scored.sort(reverse=True, key=lambda x: x[0])
    best_count, best_folder = scored[0]
    if best_count == 0:
        raise FileNotFoundError(f"No {label} embeddings found. Checked: {[str(c) for c in choices]}")
    if best_folder != configured_folder:
        configured_count = next(count for count, folder in scored if folder == configured_folder)
        print(f"[info] Using {label} embeddings from {best_folder} ({best_count} files matched).")
        print(f"[info] Configured folder had {configured_count} matches.")
    return best_folder


def load_class_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    id_to_class = {int(v): k for k, v in class_dict.items()}
    return class_dict, id_to_class


def top_class(class_name):
    return class_name.split("-")[0]


class EmbeddingDataset(Dataset):
    def __init__(self, dataframe, audio_folder, text_folder):
        self.df = dataframe.reset_index(drop=True)
        self.audio_folder = Path(audio_folder)
        self.text_folder = Path(text_folder)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sound_id = str(row["sound_id"])
        audio = torch.tensor(np.load(self.audio_folder / f"{sound_id}.npy"), dtype=torch.float32)
        text = torch.tensor(np.load(self.text_folder / f"{sound_id}.npy"), dtype=torch.float32)
        return {
            "sound_id": sound_id,
            "audio_embedding": audio,
            "text_embedding": text,
        }


def build_scoring_df(metadata, audio_folder, text_folder):
    rows = []
    for _, row in metadata.iterrows():
        sound_id = str(row["sound_id"])
        if (audio_folder / f"{sound_id}.npy").is_file() and (text_folder / f"{sound_id}.npy").is_file():
            rows.append(row)
    if not rows:
        raise RuntimeError("No BSD35k rows had both audio and text embeddings.")
    return pd.DataFrame(rows).reset_index(drop=True)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = BaseClassifier(
        hidden_size=cfg.get("hidden_size", 128),
        num_classes=cfg["num_classes"],
        emb_size_audio=cfg.get("emb_size_audio", 512),
        emb_size_text=cfg.get("emb_size_text", 512),
        dropout=cfg.get("dropout", 0.1),
        use_batch_norm=cfg.get("use_batch_norm", True),
        mode=cfg.get("mode", "both"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_ensemble(model_paths, loader, device):
    all_probs = []
    attention_audio = []
    attention_text = []
    for model_path in model_paths:
        print(f"[step] Predicting with {model_path}")
        model = load_model(model_path, device)
        fold_probs = []
        fold_attn_audio = []
        fold_attn_text = []
        with torch.no_grad():
            for batch in loader:
                audio = batch["audio_embedding"].to(device)
                text = batch["text_embedding"].to(device)
                _, logits, attn = model(audio, text)
                fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                if attn is not None:
                    fold_attn_audio.append(attn[:, 0].cpu().numpy())
                    fold_attn_text.append(attn[:, 1].cpu().numpy())
        all_probs.append(np.vstack(fold_probs))
        if fold_attn_audio:
            attention_audio.append(np.concatenate(fold_attn_audio))
            attention_text.append(np.concatenate(fold_attn_text))

    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    disagreement = np.std(np.stack(all_probs, axis=0), axis=0).mean(axis=1)
    if attention_audio:
        attn_audio = np.mean(np.stack(attention_audio, axis=0), axis=0)
        attn_text = np.mean(np.stack(attention_text, axis=0), axis=0)
    else:
        attn_audio = None
        attn_text = None
    return probs, disagreement, attn_audio, attn_text


def main():
    parser = argparse.ArgumentParser(description="Score BSD35k-CS with trained BSD10k classifier checkpoints.")
    parser.add_argument("--model-dir", default="dcase2026_task1_baseline/model_output/both")
    parser.add_argument("--output-dir", default="experiments/bsd35k_scoring")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config()
    bsd35k_cfg = config["datasets"]["BSD35k-CS"]
    metadata = filter_target_classes(pd.read_csv(resolve_path(bsd35k_cfg["metadata_csv"])))

    audio_folder = pick_embedding_folder(
        metadata,
        bsd35k_cfg["audio_emb_folder"],
        ["data/features/BSD35k_clap_audio_embeddings"],
        "BSD35k audio",
    )
    text_folder = pick_embedding_folder(
        metadata,
        bsd35k_cfg["text_emb_folder"],
        ["data/features/BSD35k-CS_clap_text_embeddings"],
        "BSD35k text",
    )

    class_dict_path = BASELINE_DIR / "data" / "class_dict.json"
    if not class_dict_path.exists():
        raise FileNotFoundError(f"Missing class dictionary: {class_dict_path}")
    class_dict, id_to_class = load_class_dict(class_dict_path)
    class_names = [id_to_class[i] for i in range(len(id_to_class))]

    model_dir = resolve_path(args.model_dir)
    model_paths = sorted(model_dir.glob("fold_*/best_model.pth"))
    if not model_paths:
        raise FileNotFoundError(f"No fold checkpoints found under {model_dir}")
    print(f"[info] Found {len(model_paths)} checkpoints.")

    scoring_df = build_scoring_df(metadata, audio_folder, text_folder)
    print(f"[info] BSD35k usable rows: {len(scoring_df):,} / {len(metadata):,}")

    dataset = EmbeddingDataset(scoring_df, audio_folder, text_folder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    probs, disagreement, attn_audio, attn_text = predict_ensemble(model_paths, loader, device)

    pred_idx = probs.argmax(axis=1)
    provided_idx = scoring_df["class"].map(class_dict)
    provided_prob = np.array(
        [
            probs[row_idx, int(cls_idx)] if pd.notna(cls_idx) else np.nan
            for row_idx, cls_idx in enumerate(provided_idx)
        ]
    )

    out = scoring_df.copy()
    out["provided_class"] = out["class"]
    out["provided_top_class"] = out["class_top"]
    out["predicted_class"] = [id_to_class[int(i)] for i in pred_idx]
    out["predicted_top_class"] = out["predicted_class"].map(top_class)
    out["classifier_confidence"] = probs.max(axis=1)
    out["provided_class_probability"] = provided_prob
    out["classifier_margin"] = np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]
    out["ensemble_disagreement"] = disagreement
    out["same_class"] = out["provided_class"] == out["predicted_class"]
    out["same_top_class"] = out["provided_top_class"] == out["predicted_top_class"]
    if attn_audio is not None:
        out["attention_audio"] = attn_audio
        out["attention_text"] = attn_text

    for idx, class_name in enumerate(class_names):
        out[f"prob_{class_name}"] = probs[:, idx]

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "BSD35k-CS_classifier_scores.csv"
    out.to_csv(output_csv, index=False)

    summary = {
        "model_dir": str(model_dir),
        "checkpoints": [str(p) for p in model_paths],
        "rows_scored": int(len(out)),
        "same_class_rate": float(out["same_class"].mean()),
        "same_top_class_rate": float(out["same_top_class"].mean()),
        "mean_classifier_confidence": float(out["classifier_confidence"].mean()),
        "mean_provided_class_probability": float(np.nanmean(provided_prob)),
    }
    with open(output_dir / "classifier_scoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    by_class = (
        out.groupby(["provided_top_class", "provided_class"], observed=True)
        .agg(
            n=("sound_id", "count"),
            same_class_rate=("same_class", "mean"),
            same_top_class_rate=("same_top_class", "mean"),
            mean_classifier_confidence=("classifier_confidence", "mean"),
            mean_provided_class_probability=("provided_class_probability", "mean"),
            mean_ensemble_disagreement=("ensemble_disagreement", "mean"),
        )
        .reset_index()
        .sort_values(["same_class_rate", "n"], ascending=[True, False])
    )
    by_class.to_csv(output_dir / "BSD35k_classifier_scores_by_class.csv", index=False)

    by_uploader = (
        out.groupby("uploader", observed=True)
        .agg(
            n=("sound_id", "count"),
            same_class_rate=("same_class", "mean"),
            same_top_class_rate=("same_top_class", "mean"),
            mean_classifier_confidence=("classifier_confidence", "mean"),
            mean_provided_class_probability=("provided_class_probability", "mean"),
            dominant_class=("provided_class", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
        .sort_values(["same_class_rate", "n"], ascending=[True, False])
    )
    by_uploader.to_csv(output_dir / "BSD35k_classifier_scores_by_uploader.csv", index=False)

    print(f"[done] Scores saved to {output_csv}")
    print(
        "[summary] same_class_rate={:.3f}, same_top_class_rate={:.3f}, mean_conf={:.3f}".format(
            summary["same_class_rate"],
            summary["same_top_class_rate"],
            summary["mean_classifier_confidence"],
        )
    )


if __name__ == "__main__":
    main()
