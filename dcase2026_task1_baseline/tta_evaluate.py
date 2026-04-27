"""TTA (Test-Time Augmentation) evaluator: re-evaluates an existing 5-fold experiment
by adding Gaussian noise N times to embeddings and averaging softmax outputs.

No retraining required. Uses already-saved best_model.pth + splits.csv per fold.

Usage:
    python tta_evaluate.py --exp_name exp_006_conf4_hier_h256 --n_aug 5 --noise_std 0.001
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import get_subconfig, build_class_to_topclass_mapping, build_id_to_class_mapping
from models import BaseClassifier
from dataset_utils import HATRDataset
from evaluate import hierarchical_accuracy, hierarchical_prf_weighted
from utils import extend_subcat, get_top_level, intersection


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True,
                   help="Experiment name under model_output/ (e.g. 'exp_006_conf4_hier_h256')")
    p.add_argument('--mode', type=str, default='both', choices=['both', 'audio', 'text'])
    p.add_argument('--n_aug', type=int, default=5)
    p.add_argument('--noise_std', type=float, default=1e-3,
                   help='Gaussian noise std added to embeddings at inference (default 1e-3, 10x training)')
    p.add_argument('--out', type=str, default=None,
                   help='Output JSON. Default: experiments/exp_009_tta_<exp_name>.json')
    return p.parse_args()


def predict_with_tta(model, dataset, device, batch_size, n_aug, noise_std):
    model.eval()
    n = len(dataset)
    all_probs = None
    sids, gts = [], []
    for aug_i in range(n_aug):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
        probs_run = []
        if aug_i == 0:
            sids = []; gts = []
        with torch.no_grad():
            for data in loader:
                y = data['class_idx'].to(device)
                audio = data['audio_embedding'].to(device)
                text = data['text_embedding'].to(device)
                if aug_i > 0:
                    audio = audio + torch.randn_like(audio) * noise_std
                    text = text + torch.randn_like(text) * noise_std
                _, logits, _ = model(audio, text)
                probs_run.append(torch.softmax(logits, dim=1).cpu().numpy())
                if aug_i == 0:
                    for i in range(y.size(0)):
                        sid = data['sound_id'][i]
                        if isinstance(sid, torch.Tensor): sid = sid.item()
                        sids.append(sid); gts.append(y[i].item())
        probs_run = np.concatenate(probs_run, axis=0)
        all_probs = probs_run if all_probs is None else all_probs + probs_run
    avg_probs = all_probs / n_aug
    preds = avg_probs.argmax(axis=1).tolist()
    scores = avg_probs.max(axis=1).tolist()
    return sids, gts, preds, scores


def compute_metrics(gts, preds, class_dict, class_to_topclass):
    id2c = build_id_to_class_mapping(class_dict)
    pl = [id2c.get(p, str(p)) for p in preds]
    gl = [id2c.get(g, str(g)) for g in gts]
    pairs = list(zip(pl, gl)); classes = list(set(gl))
    total = len(gts)
    correct = sum(p == g for p, g in zip(preds, gts))
    top_correct = sum(class_to_topclass.get(g) == class_to_topclass.get(p) for p, g in zip(preds, gts))
    macro_a, macro_t = [], []
    for c in classes:
        idx = [i for i, g in enumerate(gl) if g == c]
        if not idx: continue
        macro_a.append(sum(1 for i in idx if preds[i] == gts[i]) / len(idx))
        macro_t.append(sum(1 for i in idx if class_to_topclass.get(gts[i]) == class_to_topclass.get(preds[i])) / len(idx))
    h_accs = []
    for c in classes:
        try:
            v = hierarchical_accuracy(c, pairs, 0.5)
            if not np.isnan(v): h_accs.append(v)
        except Exception: pass
    hPs, hRs, hFs = [], [], []
    for c in classes:
        try:
            p, r, f = hierarchical_prf_weighted(c, pairs, 0.75)
            if not (np.isnan(p) or np.isnan(r) or np.isnan(f)):
                hPs.append(p); hRs.append(r); hFs.append(f)
        except Exception: pass
    return {
        'accuracy': 100*correct/total if total else 0,
        'top_accuracy': 100*top_correct/total if total else 0,
        'macro_accuracy': 100*np.mean(macro_a) if macro_a else 0,
        'macro_top_accuracy': 100*np.mean(macro_t) if macro_t else 0,
        'hierarchical_accuracy': 100*np.mean(h_accs) if h_accs else 0,
        'hierarchical_precision': 100*np.mean(hPs) if hPs else 0,
        'hierarchical_recall': 100*np.mean(hRs) if hRs else 0,
        'hierarchical_f1': 100*np.mean(hFs) if hFs else 0,
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = get_subconfig('output_path')
    prepared_csv = os.path.join(data_dir, get_subconfig('processed_dataset_csv'))
    class_dict_json = os.path.join(data_dir, get_subconfig('class_dict_json'))
    top_class_dict_json = os.path.join(data_dir, get_subconfig('top_class_dict_json'))

    with open(class_dict_json) as f: class_dict = json.load(f)
    with open(top_class_dict_json) as f: top_class_dict = json.load(f)
    class_to_topclass = build_class_to_topclass_mapping(class_dict, top_class_dict)

    full_df = pd.read_csv(prepared_csv)
    full_df['index'] = full_df['index'].astype(str)

    fold_metrics = []
    base = os.path.join('./model_output', args.exp_name, args.mode)
    for fold in range(5):
        fold_dir = os.path.join(base, f'fold_{fold}')
        ckpt_path = os.path.join(fold_dir, 'best_model.pth')
        splits_path = os.path.join(fold_dir, 'splits.csv')
        if not (os.path.isfile(ckpt_path) and os.path.isfile(splits_path)):
            print(f"  [skip fold {fold}] missing checkpoint or splits")
            continue
        splits = pd.read_csv(splits_path)
        splits['index'] = splits['index'].astype(str)
        test_ids = set(splits[splits['split'] == 'test']['index'].tolist())
        test_df = full_df[full_df['index'].isin(test_ids)].reset_index(drop=True)
        test_dataset = HATRDataset(test_df, aug=False)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BaseClassifier(**ckpt['config']).to(device)
        model.load_state_dict(ckpt['model_state'])

        sids, gts, preds, scores = predict_with_tta(
            model, test_dataset, device, batch_size=64,
            n_aug=args.n_aug, noise_std=args.noise_std,
        )
        m = compute_metrics(gts, preds, class_dict, class_to_topclass)
        print(f"  fold {fold} -> H-Acc={m['hierarchical_accuracy']:.2f}% "
              f"acc={m['accuracy']:.2f}% top={m['top_accuracy']:.2f}%")
        fold_metrics.append({'fold': fold, **m})

    keys = ['accuracy','top_accuracy','macro_accuracy','macro_top_accuracy',
            'hierarchical_accuracy','hierarchical_precision','hierarchical_recall','hierarchical_f1']
    avg = {k: float(np.mean([r[k] for r in fold_metrics])) for k in keys}
    std = {k: float(np.std([r[k] for r in fold_metrics])) for k in keys}
    print(f"\n[TTA n={args.n_aug} std={args.noise_std}] H-Acc={avg['hierarchical_accuracy']:.2f}% "
          f"+- {std['hierarchical_accuracy']:.2f}% (over {len(fold_metrics)} folds)")

    out = args.out or os.path.join('..', 'experiments', f'exp_009_tta_{args.exp_name}.json')
    payload = {
        'exp_id': 'exp_009',
        'description': f'TTA (Gaussian noise std={args.noise_std}, n_aug={args.n_aug}) on top of {args.exp_name}',
        'base_exp': args.exp_name,
        'config': {
            'n_aug': args.n_aug, 'noise_std': args.noise_std, 'mode': args.mode,
        },
        'results': {
            'fold_avg': {k: round(avg[k]/100, 6) for k in keys},
            'fold_std': {k: round(std[k]/100, 6) for k in keys},
            'fold_details': [{'fold': r['fold'], **{k: round(r[k]/100, 6) for k in keys}} for r in fold_metrics],
        },
        'vs_baseline_h_acc': f'+{avg["hierarchical_accuracy"]-79.45:.2f}%',
    }
    with open(out, 'w') as f: json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
