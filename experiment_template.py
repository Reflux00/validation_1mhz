"""Frozen raw-spectrum nearest-centroid experiment; no test-driven selection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestCentroid
from threadpoolctl import threadpool_limits

from THz_data import load_all_data


def metrics(true, predicted):
    error = predicted - true  # Filename suffixes are integer MHz offsets.
    return {
        'accuracy': float(accuracy_score(true, predicted)),
        'f1_macro': float(f1_score(true, predicted, average='macro')),
        'correct_samples': int((true == predicted).sum()), 'num_samples': len(true),
        'mae_mhz': float(np.abs(error).mean()),
        'rmse_mhz': float(np.sqrt(np.mean(error.astype(float) ** 2))),
        'max_error_mhz': int(np.abs(error).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/20261003_optimization/final_template'))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'evaluation_metrics.json').exists():
        raise FileExistsError('Final evaluation exists; choose a new output directory')
    root = Path(__file__).resolve().parent
    previous = root / 'runs/20261003_independent_holdout'
    indices = dict(np.load(previous / 'split_indices.npz'))
    train, val, test = (indices[k] for k in ('train', 'val', 'test'))
    assert np.array_equal(np.sort(np.concatenate([train, val, test])), np.arange(1100))
    labels, spectra, paths = load_all_data(return_paths=True)
    np.save(out / 'spectral_axis.npy', np.loadtxt(paths[train[0]])[:, 0])
    selection = {
        'algorithm': 'NearestCentroid', 'metric': 'euclidean', 'priors': 'uniform',
        'input': 'All 15365 raw intensity values; no file metadata, masking, normalization or augmentation',
        'selection': 'First baseline reached 220/220 validation accuracy; frozen before this final test',
        'training_samples': len(train), 'validation_samples': len(val), 'test_samples': len(test),
        'final_training_indices': 'Same 660 original training spectra as the previous CNN',
        'cv': 'Four stratified folds within the original train+validation pool (880); excludes all 220 test spectra',
        'cv_seed': 44,
        'scope': 'Known-frequency repeated-measurement classification. The final test is the previously reported holdout, not a new external dataset.',
    }
    (out / 'selection_frozen.json').write_text(json.dumps(selection, indent=2) + '\n')
    np.savez(out / 'split_indices.npz', **indices)
    shutil.copyfile(previous / 'split_manifest.csv', out / 'split_manifest.csv')
    (out / 'source').mkdir(exist_ok=True)
    for name in ('experiment_template.py', 'predict_template.py', 'THz_data.py'):
        shutil.copyfile(root / name, out / 'source' / name)

    with threadpool_limits(limits=4):
        # The method is already frozen; CV describes stability, not model selection.
        development = np.r_[train, val]
        folds, rows = [], []
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=44)
        for fold, (fit_rows, held_rows) in enumerate(cv.split(development, labels[development]), start=1):
            fit_idx, held_idx = development[fit_rows], development[held_rows]
            assert set(fit_idx).isdisjoint(test) and set(held_idx).isdisjoint(test)
            model = NearestCentroid().fit(spectra[fit_idx], labels[fit_idx])
            pred = model.predict(spectra[held_idx])
            fold_metrics = {'fold': fold, **metrics(labels[held_idx], pred)}
            folds.append(fold_metrics)
            rows.append(pd.DataFrame({'fold': fold, 'source_id': held_idx,
                                      'true_label': labels[held_idx], 'pred_label': pred}))
            print('Development CV:', json.dumps(fold_metrics), flush=True)
        pd.DataFrame(folds).to_csv(out / 'cross_validation_metrics.csv', index=False)
        pd.concat(rows, ignore_index=True).to_csv(out / 'cross_validation_predictions.csv', index=False)

        # Keep exactly the original 660 training spectra for the CNN comparison.
        started = time.perf_counter()
        model = NearestCentroid().fit(spectra[train], labels[train])
        fit_seconds = time.perf_counter() - started
        centers = np.array([spectra[train][labels[train] == c].mean(axis=0) for c in model.classes_])
        np.testing.assert_allclose(model.centroids_, centers)
        val_pred = model.predict(spectra[val])
        assert (val_pred == labels[val]).all(), 'Frozen validation result was not reproduced'
        joblib.dump(model, out / 'model.joblib')
        restored = joblib.load(out / 'model.joblib')
        np.testing.assert_array_equal(restored.predict(spectra[val]), val_pred)

        # Exactly one final test prediction after method and training set are fixed.
        started = time.perf_counter()
        pred = restored.predict(spectra[test])
        predict_seconds = time.perf_counter() - started
    result = metrics(labels[test], pred)
    result.update({'model': 'raw_spectrum_nearest_centroid', 'training_samples': len(train),
                   'fit_seconds': fit_seconds, 'test_predict_seconds': predict_seconds,
                   'test_training_source_overlap': 0,
                   'cross_validation_accuracy_mean': float(np.mean([f['accuracy'] for f in folds])),
                   'cross_validation_accuracy_std': float(np.std([f['accuracy'] for f in folds])),
                   'scope': selection['scope']})
    predictions = pd.DataFrame({
        'source_id': test, 'source_file': [str(paths[i].relative_to(root)) for i in test],
        'repeat_id': [int(paths[i].stem.rsplit('_', 1)[1]) for i in test],
        'true_label': labels[test], 'pred_label': pred,
        'true_frequency_ghz': (548000 + labels[test]) / 1000,
        'predicted_frequency_ghz': (548000 + pred) / 1000,
        'absolute_error_mhz': np.abs(labels[test] - pred), 'is_correct': (labels[test] == pred).astype(int),
    })
    predictions.to_csv(out / 'prediction_results_frequency.csv', index=False)
    predictions.loc[predictions.is_correct == 0].to_csv(out / 'misclassified_samples.csv', index=False)
    (out / 'evaluation_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    old = json.loads((previous / 'evaluation_metrics.json').read_text())
    pd.DataFrame([
        {'model': 'Original CNN', 'original_train_samples': 660, **{k: old[k] for k in ('accuracy', 'f1_macro', 'mae_mhz', 'rmse_mhz', 'max_error_mhz')}},
        {'model': 'Raw-spectrum nearest centroid', 'original_train_samples': 660, **{k: result[k] for k in ('accuracy', 'f1_macro', 'mae_mhz', 'rmse_mhz', 'max_error_mhz')}},
    ]).to_csv(out / 'comparison.csv', index=False)
    (out / 'model_sha256.txt').write_text(hashlib.sha256((out / 'model.joblib').read_bytes()).hexdigest() + '\n')
    print('Final test:', json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
