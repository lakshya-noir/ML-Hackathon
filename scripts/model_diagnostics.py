"""Reconstruct binary candidate confusion matrices from retained baseline rows.

These are candidate-level diagnostics against nearest-route surrogate labels,
not observation Top-1 or final HMM accuracy. No model is fitted or changed.
"""
import csv
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache_path = ROOT / 'outputs/features-200.joblib'
    if not cache_path.exists():
        raise SystemExit('Missing original feature cache. See docs/REPRODUCING.md; recreate in a separate checkout.')
    cache = joblib.load(cache_path)
    comparison = json.loads((ROOT / 'results/model_comparison.json').read_text())
    names = list(comparison['models']) + ['tuned']
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for name, ax in zip(names, axes.flat):
        model = joblib.load(ROOT / 'artifacts/models' / name / 'model.joblib')['model']
        for partition in ('train', 'student_test'):
            features, truth = [], []
            for record in cache[partition]:
                for candidates, label in record['observations']:
                    for candidate in candidates:
                        features.append(candidate['features'])
                        truth.append(int(candidate['arc_id'] == label))
            prediction = model.predict_proba(np.asarray(features))[:, 1] >= .5
            matrix = confusion_matrix(truth, prediction, labels=[0, 1])
            tn, fp, fn, tp = map(int, matrix.ravel())
            rows.append(dict(model=name, partition=partition, threshold=.5,
                             candidate_rows=len(truth), tn=tn, fp=fp, fn=fn, tp=tp))
            if partition == 'student_test':
                ax.imshow(matrix, cmap='Blues')
                for (i, j), value in np.ndenumerate(matrix):
                    ax.text(j, i, f'{value:,}', ha='center', va='center',
                            color='white' if value > matrix.max()/2 else 'black')
                ax.set(title=name.replace('_', ' '), xlabel='Predicted', ylabel='Surrogate label',
                       xticks=[0, 1], yticks=[0, 1])
        print(name, 'diagnostics complete', flush=True)
    fig.suptitle('Student-test candidate confusion matrices | threshold 0.5 | original sampled experiment')
    fig.tight_layout()
    out = ROOT / 'results'
    with (out / 'confusion_matrices.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'confusion_matrices.json').write_text(json.dumps({
        'scope': 'candidate binary classification; original 200-observation-per-trip experiment',
        'label_method': comparison['label_method'], 'no_refitting': True, 'rows': rows}, indent=2)+'\n')
    fig.savefig(out / 'confusion_matrices.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
