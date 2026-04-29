import argparse
import collections.abc
from collections import defaultdict
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold,
    GroupKFold, StratifiedGroupKFold, GroupShuffleSplit,
)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from losses import CrossEntropyLoss, HierarchicalLoss
from utils import (
    get_subconfig,
    set_seed,
    build_class_to_topclass_mapping,
    build_class_to_topclass_tensor,
)
from models import BaseClassifier
from dataset_utils import HATRDataset
from evaluate import evaluate_model

# Paths
dataset_name = get_subconfig("active_dataset")
dataset_path = get_subconfig("datasets")[dataset_name]["metadata_csv"]

data_dir = get_subconfig("output_path")
prepared_dataset_path = os.path.join(data_dir, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))
subclass_json = os.path.join(data_dir, get_subconfig("top_class_subclass_dict_json"))


def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)


def make_serializable(obj, decimals=6):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        return make_serializable(obj, decimals)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return round(float(obj), decimals)
        else:
            return [make_serializable(x, decimals) for x in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return {k: make_serializable(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return [make_serializable(x, decimals) for x in obj]
    else:
        return obj


def train_model(model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, classification_weight=1.0, classification_criterion=None,
                output_dir='model_output', scheduler_type='plateau', patience=10, early_stopping_factor=5):
    """Train a model. Supports both plain CE and HierarchicalLoss criterions
    (the hier criterion expects logits, labels, z, top_labels and returns (loss, components))."""

    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, verbose=True)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    best_accuracy = 0.0
    epochs_without_improvement = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        losses = defaultdict(float)
        total_samples = 0

        attn_audio_epoch = []
        attn_text_epoch = []

        for data in train_loader:
            class_labels = data['class_idx'].to(device)
            top_labels = data['top_class_idx'].to(device) if 'top_class_idx' in data else None
            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)

            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            optimizer.zero_grad()

            z, class_logit, attn_scores = model(audio_emb, text_emb)

            if attn_scores is not None:
                attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
                attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

            total_loss = 0.0
            batch_size = class_labels.size(0)
            total_samples += batch_size

            if classification_criterion is not None:
                cls_loss, components = classification_criterion(
                    class_logit, class_labels, z=z, top_labels=top_labels
                )
                losses['cls'] += cls_loss.item() * batch_size
                if 'ce' in components:
                    losses['ce'] += float(components['ce']) * batch_size
                if 'top' in components:
                    losses['top'] += float(components['top']) * batch_size
                if 'contr' in components:
                    losses['contr'] += float(components['contr']) * batch_size
                total_loss = total_loss + classification_weight * cls_loss

            total_loss.backward()
            optimizer.step()
            losses['total'] += total_loss.item() * batch_size

        if attn_audio_epoch:
            attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
            attn_text_epoch = torch.cat(attn_text_epoch, dim=0)
            history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
            history["attention_text"].append(attn_text_epoch.mean(0).numpy())

        for k in losses:
            history[f'train_{k}_loss'].append(losses[k] / total_samples)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                labels = data['class_idx'].to(device)
                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)

                if audio_emb is not None:
                    audio_emb = audio_emb.to(device)
                if text_emb is not None:
                    text_emb = text_emb.to(device)

                _, class_logit, _ = model(audio_emb, text_emb)

                _, predicted = torch.max(class_logit.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        history['val_accuracy'].append(val_accuracy)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        loss_summary = " | ".join([f"{k}:{losses[k]/total_samples:.4f}" for k in ('total', 'ce', 'top', 'contr') if k in losses])
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Val acc: {val_accuracy:.2f}% | {loss_summary}")

        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_config = {
                'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else 128,
                'num_classes': model.num_classes if hasattr(model, 'num_classes') else None,
                'emb_size_audio': model.emb_size_audio if hasattr(model, 'emb_size_audio') else 0,
                'emb_size_text': model.emb_size_text if hasattr(model, 'emb_size_text') else 0,
                'dropout': model.dropout if hasattr(model, 'dropout') else 0.1,
                'use_batch_norm': True,
                'mode': model.mode if hasattr(model, 'mode') else 'both',
            }

            torch.save({
                'model_state': model.state_dict(),
                'config': model_config,
            }, os.path.join(output_dir, "best_model.pth"))

            print(f"  New best model saved")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience * early_stopping_factor:
                print("Early stopping triggered.")
                break

    return best_accuracy, history, model


def parse_args():
    p = argparse.ArgumentParser(description="DCASE 2026 Task 1 trainer")
    p.add_argument('--exp_name', type=str, default='baseline',
                   help='Experiment name. Output goes to model_output/<exp_name>/<mode>/fold_*')
    p.add_argument('--modes', type=str, nargs='+', default=['both', 'audio'],
                   choices=['both', 'audio', 'text'],
                   help='Which modalities to train. Default: both audio')
    p.add_argument('--conf_threshold', type=int, default=None,
                   help='Filter samples with confidence < threshold. None = no filter.')
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--k_folds', type=int, default=5)
    p.add_argument('--scheduler', type=str, default='step', choices=['step', 'plateau', 'none'])
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--early_stopping_factor', type=int, default=3)
    p.add_argument('--seed', type=int, default=1821, help='Default 1821 to match EXP-000 baseline.')

    p.add_argument('--hier_loss', action='store_true',
                   help='Use HierarchicalLoss (L_CE + lambda_top*L_Top + lambda_contr*L_Contr)')
    p.add_argument('--lambda_top', type=float, default=0.3)
    p.add_argument('--lambda_contr', type=float, default=0.1)
    p.add_argument('--tau', type=float, default=0.07)

    p.add_argument('--class_weights', type=str, default='none', choices=['none', 'balanced', 'sqrt'],
                   help="Per-class weights for CE: 'balanced' = N/(K*count), 'sqrt' = sqrt(N/count) (less aggressive)")

    p.add_argument('--fold_strategy', type=str, default='random',
                   choices=['random', 'group', 'stratified_group'],
                   help="Cross-validation split strategy. 'random'=StratifiedKFold by class (default, EXP-008). "
                        "'group'=GroupKFold by uploader. 'stratified_group'=StratifiedGroupKFold (recommended for private LB proxy).")

    p.add_argument('--smoke_test', action='store_true',
                   help='Run only fold 0 with epochs=2 to validate the pipeline.')
    return p.parse_args()


def main():
    args = parse_args()
    seed = set_seed(args.seed)
    print(f"[config] exp={args.exp_name} mode={args.modes} conf>={args.conf_threshold} "
          f"hidden={args.hidden_size} dropout={args.dropout} hier_loss={args.hier_loss} "
          f"lambda_top={args.lambda_top} lambda_contr={args.lambda_contr} tau={args.tau} seed={seed}")

    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)

    model_output = os.path.join('./model_output', args.exp_name)

    full_df = pd.read_csv(prepared_dataset_path)

    high_conf_ids = None
    if args.conf_threshold is not None:
        conf_df = pd.read_csv(dataset_path)
        conf_df['sound_id'] = conf_df['sound_id'].astype(str).str.strip()
        high_conf_ids = set(conf_df.loc[conf_df['confidence'] >= args.conf_threshold, 'sound_id'])
        full_df['index'] = full_df['index'].astype(str)
        n_full_high_conf = int(full_df['index'].isin(high_conf_ids).sum())
        print(f"[conf-filter PREP] threshold>={args.conf_threshold} | "
              f"{len(high_conf_ids)} high-conf sound_ids loaded "
              f"({n_full_high_conf}/{len(full_df)} samples in dataset). "
              f"Will be applied to TRAIN ONLY in fold loop; val/test keep all samples.")

    if args.fold_strategy in ('group', 'stratified_group'):
        meta_df = pd.read_csv(dataset_path)
        meta_df['sound_id'] = meta_df['sound_id'].astype(str).str.strip()
        full_df['index'] = full_df['index'].astype(str).str.strip()
        full_df = full_df.merge(
            meta_df[['sound_id', 'uploader']],
            left_on='index', right_on='sound_id', how='left'
        )
        n_missing = int(full_df['uploader'].isna().sum())
        if n_missing > 0:
            print(f"[fold-strategy] WARNING: {n_missing} samples missing uploader -> using sound_id as fallback group.")
            full_df['uploader'] = full_df['uploader'].fillna(full_df['index'])
        full_df = full_df.drop(columns=['sound_id'])
        print(f"[fold-strategy={args.fold_strategy}] joined uploader column "
              f"({full_df['uploader'].nunique()} unique uploaders, {len(full_df)} samples)")

    epochs = 2 if args.smoke_test else args.epochs

    datasets = {f'{dataset_name} full': {'df': full_df}}

    fold_metrics_all = []

    for dataset, dataset_info in datasets.items():
        print(f"\n=== Dataset: {dataset} ===")
        database = dataset_info['df']
        labels = database["class_idx"].tolist()
        n_splits = max(args.k_folds, 2)

        groups = None
        if args.fold_strategy in ('group', 'stratified_group'):
            groups = database['uploader'].values

        if args.fold_strategy == 'random':
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            all_splits = list(skf.split(np.zeros(len(labels)), labels))
        elif args.fold_strategy == 'group':
            gkf = GroupKFold(n_splits=n_splits)
            all_splits = list(gkf.split(np.zeros(len(labels)), labels, groups=groups))
        elif args.fold_strategy == 'stratified_group':
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            all_splits = list(sgkf.split(np.zeros(len(labels)), labels, groups=groups))
        else:
            raise ValueError(f"Unknown fold_strategy: {args.fold_strategy}")

        for fi, (tr_idx, te_idx) in enumerate(all_splits):
            tr_classes = len(np.unique([labels[i] for i in tr_idx]))
            te_classes = len(np.unique([labels[i] for i in te_idx]))
            extra = ""
            if groups is not None:
                tr_groups = len(np.unique(groups[tr_idx]))
                te_groups = len(np.unique(groups[te_idx]))
                overlap = len(set(groups[tr_idx]) & set(groups[te_idx]))
                extra = f" | uploaders tr={tr_groups} te={te_groups} overlap={overlap}"
            print(f"  [fold {fi}] train={len(tr_idx)} (cls={tr_classes}) test={len(te_idx)} (cls={te_classes}){extra}")

        if args.smoke_test:
            all_splits = all_splits[:1]

        for mode in args.modes:
            print(f"\n=== Running experiments: Dataset={dataset} | Mode={mode} ===")

            for fold, (trainval_idx, test_idx) in enumerate(all_splits):
                print(f"\n==== Fold {fold} ====")

                trainval_labels = [labels[i] for i in trainval_idx]
                if args.fold_strategy == 'random':
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
                else:
                    trainval_groups = groups[trainval_idx]
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    train_idx_rel, val_idx_rel = next(gss.split(np.zeros(len(trainval_labels)), trainval_labels, groups=trainval_groups))
                train_idx = [trainval_idx[i] for i in train_idx_rel]
                val_idx = [trainval_idx[i] for i in val_idx_rel]

                train_df = database.iloc[train_idx].reset_index(drop=True)
                val_df = database.iloc[val_idx].reset_index(drop=True)
                test_df = database.iloc[test_idx].reset_index(drop=True)

                if high_conf_ids is not None:
                    tr_before = len(train_df)
                    train_df = train_df[train_df['index'].astype(str).isin(high_conf_ids)].reset_index(drop=True)
                    print(f"[Fold {fold}] Train: {tr_before} -> {len(train_df)} (conf>={args.conf_threshold} 필터, train만)")
                    print(f"           Val:   {len(val_df)} (필터 없음, 모두 유지)")
                    print(f"           Test:  {len(test_df)} (필터 없음, 모두 유지)")
                    if len(train_df) == 0:
                        raise RuntimeError(f"Fold {fold} train set empty after conf filter — threshold too strict")
                else:
                    print(f"[Fold {fold}] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} (no conf filter)")

                train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.7)
                val_dataset = HATRDataset(val_df, aug=False)
                test_dataset = HATRDataset(test_df, aug=False)

                train_loader = DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=0, pin_memory=torch.cuda.is_available())
                val_loader = DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=torch.cuda.is_available())
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=torch.cuda.is_available())

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                emb_size_audio = 512 if mode in ['audio', 'both'] else 0
                emb_size_text = 512 if mode in ['text', 'both'] else 0

                hidden_size = args.hidden_size
                dropout = args.dropout
                use_batch_norm = True

                model = BaseClassifier(
                    hidden_size=hidden_size,
                    num_classes=len(class_dict),
                    emb_size_audio=emb_size_audio,
                    emb_size_text=emb_size_text,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    mode=mode,
                ).to(device)

                cw_tensor = None
                if args.class_weights != 'none':
                    counts = np.bincount(train_df['class_idx'].values, minlength=len(class_dict)).astype(float)
                    counts = np.where(counts == 0, 1.0, counts)
                    if args.class_weights == 'balanced':
                        w = counts.sum() / (len(class_dict) * counts)
                    elif args.class_weights == 'sqrt':
                        w = np.sqrt(counts.sum() / counts)
                    w = w * (len(class_dict) / w.sum())
                    cw_tensor = torch.tensor(w, dtype=torch.float32, device=device)
                    print(f"  [class_weights={args.class_weights}] min={w.min():.3f} max={w.max():.3f} mean={w.mean():.3f}")

                if args.hier_loss:
                    sub2top_tensor = build_class_to_topclass_tensor(class_dict, top_class_dict, device)
                    classification_criterion = HierarchicalLoss(
                        subclass_to_topclass_tensor=sub2top_tensor,
                        num_top_classes=len(top_class_dict),
                        lambda_top=args.lambda_top,
                        lambda_contr=args.lambda_contr,
                        tau=args.tau,
                        class_weights=cw_tensor,
                    ).to(device)
                else:
                    classification_criterion = CrossEntropyLoss(class_weights=cw_tensor).to(device)

                output_dir = os.path.join(model_output, mode, f"fold_{fold}")
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, "best_model.pth")

                init_weights(model)

                best_accuracy, history, trained_model = train_model(
                    model, train_loader, val_loader, device,
                    num_epochs=epochs, lr=args.lr,
                    classification_weight=1.0,
                    classification_criterion=classification_criterion,
                    output_dir=output_dir,
                    scheduler_type=args.scheduler,
                    patience=args.patience,
                    early_stopping_factor=args.early_stopping_factor,
                )
                print(f"Best validation accuracy: {best_accuracy:.2f}%")

                splits_df = pd.concat([
                    train_df[['index']].assign(split='train'),
                    val_df[['index']].assign(split='val'),
                    test_df[['index']].assign(split='test')
                ])
                splits_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)

                history['model_info'] = {
                    'model_class': trained_model.__class__.__name__,
                    'hidden_size': hidden_size,
                    'num_classes': len(class_dict),
                    'emb_size_audio': emb_size_audio,
                    'emb_size_text': emb_size_text,
                    'dropout': dropout,
                    'use_batch_norm': True,
                    'mode': mode,
                    'num_folds': args.k_folds,
                    'fold_id': fold,
                    'batch_size': args.batch_size,
                    'random_seed': seed,
                    'exp_name': args.exp_name,
                    'conf_threshold': args.conf_threshold,
                    'hier_loss': args.hier_loss,
                    'lambda_top': args.lambda_top,
                    'lambda_contr': args.lambda_contr,
                    'tau': args.tau,
                }

                history_path = os.path.join(output_dir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(make_serializable(history), f, indent=2)

                class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)
                metrics = evaluate_model(
                    BaseClassifier,
                    model_path,
                    test_loader,
                    device,
                    class_to_top_class,
                    output_dir=output_dir,
                    fold_id=fold,
                    class_dict=class_dict,
                )

                fold_metrics_all.append({'mode': mode, 'fold': fold, **metrics})

                print("\n===== Fold Results =====")
                print(f"Final model accuracy: {metrics['accuracy']:.2f}%")
                print(f"Final model top-level accuracy: {metrics['top_accuracy']:.2f}%")
                print(f"Hierarchical accuracy: {metrics['hierarchical_accuracy']:.2f}%")
                print("========================")

    # Aggregate
    print("\n========= AGGREGATE =========")
    by_mode = defaultdict(list)
    for r in fold_metrics_all:
        by_mode[r['mode']].append(r)
    for mode, rows in by_mode.items():
        keys = ['accuracy', 'top_accuracy', 'macro_accuracy', 'macro_top_accuracy',
                'hierarchical_accuracy', 'hierarchical_precision', 'hierarchical_recall', 'hierarchical_f1']
        print(f"\n[mode={mode}] runs={len(rows)}")
        for k in keys:
            vals = [r[k] for r in rows]
            print(f"  {k:30s}: {np.mean(vals):.2f}% +- {np.std(vals):.2f}%")

    summary_path = os.path.join(model_output, 'summary.json')
    os.makedirs(model_output, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            'exp_name': args.exp_name,
            'config': vars(args),
            'fold_metrics': fold_metrics_all,
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    print("All experiments done!")


if __name__ == "__main__":
    main()
