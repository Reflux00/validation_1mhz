"""Evaluate unseen original measurements at the 220 known frequencies."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from THz_data import load_all_data
from model_final import (
    SpectralDataset, SpectralWavelengthPredictor, evaluate_and_save,
    offline_augment, train_model, train_val_test_split_by_label,
)


def prepare_independent_data(spectra, labels):
    """Split originals first; all fitted preprocessing and augmentation use train only."""
    split = train_val_test_split_by_label(np.arange(len(labels)), labels, seed=42)
    indices = dict(zip(('train', 'val', 'test'), split[:3]))
    assert len(np.unique(np.concatenate(list(indices.values())))) == len(labels)
    train = spectra[indices['train']]
    intensity_sum = abs(train.sum(axis=0))
    feature_mask = intensity_sum > intensity_sum.mean() / 20
    scaler = StandardScaler().fit(train[:, feature_mask])
    x, y = {}, {}
    x_train, y_train = offline_augment(
        train[:, feature_mask], labels[indices['train']], copies=1, noise_std=0.1, seed=44,
    )
    x['train'], y['train'] = offline_augment(
        scaler.transform(x_train), y_train, copies=1, noise_std=0.02, seed=45,
    )
    for name in ('val', 'test'):
        x[name] = scaler.transform(spectra[indices[name]][:, feature_mask])
        y[name] = labels[indices[name]]
    return x, y, indices, feature_mask, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='runs/20261003_independent_holdout')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'checkpoints/best.pt').exists():
        raise FileExistsError('Choose a new output directory to preserve existing results')
    started = time.monotonic()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    raw_labels, spectra, paths = load_all_data(return_paths=True)
    frequencies, labels = np.unique(raw_labels, return_inverse=True)
    x, y, indices, mask, scaler = prepare_independent_data(spectra, labels)

    # Audit original sources, including identical data stored under different names.
    manifest = pd.DataFrame({
        'source_id': np.arange(len(labels)),
        'source_file': [str(p.relative_to(Path(__file__).resolve().parent)) for p in paths],
        'repeat_id': [int(p.stem.rsplit('_', 1)[1]) for p in paths],
        'label_index': labels, 'frequency_ghz': (548000 + raw_labels) / 1000,
        'spectrum_sha256': [hashlib.sha256(row.tobytes()).hexdigest() for row in spectra],
        'split': '',
    })
    hashes = {}
    for name, idx in indices.items():
        manifest.loc[idx, 'split'] = name
        hashes[name] = set(manifest.loc[idx, 'spectrum_sha256'])
    overlaps = {f'{a}_{b}': len(hashes[a] & hashes[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    assert not any(overlaps.values()), f'Duplicate spectra across splits: {overlaps}'
    manifest.to_csv(out / 'split_manifest.csv', index=False)
    parents = np.tile(indices['train'], 4)
    assert np.array_equal(y['train'], labels[parents])
    provenance = manifest.iloc[parents].reset_index(drop=True).copy()
    provenance.insert(0, 'training_row', np.arange(len(parents)))
    n_train = len(indices['train'])
    provenance['raw_noise_augmented'] = np.tile(np.repeat([False, True], n_train), 2)
    provenance['standardized_noise_augmented'] = np.repeat([False, True], 2 * n_train)
    provenance.to_csv(out / 'training_provenance.csv', index=False)
    np.savez(out / 'split_indices.npz', **indices)
    np.save(out / 'feature_mask.npy', mask)
    joblib.dump(scaler, out / 'scaler.joblib')
    config = {
        'protocol': 'original_measurement_holdout_within_known_frequencies',
        'mode': 'classification', 'epochs': args.epochs, 'learning_rate': 0.0015,
        'batch_size': 64, 'device': device, 'torch_numpy_seed': 44, 'split_seed': 42,
        'raw_samples': len(labels), 'raw_features': spectra.shape[1],
        'retained_features': int(mask.sum()), 'num_classes': len(frequencies),
        'original_split_counts': {k: len(v) for k, v in indices.items()},
        'effective_split_counts': {k: len(v) for k, v in y.items()},
        'feature_selection_fit': '660 original training spectra only',
        'scaler_fit': '660 original training spectra only',
        'augmentation': [
            {'split': 'train', 'stage': 'raw_reduced', 'copies': 1, 'noise_std': 0.1, 'seed': 44},
            {'split': 'train', 'stage': 'standardized', 'copies': 1, 'noise_std': 0.02, 'seed': 45},
        ],
        'cross_split_identical_spectra': overlaps,
        'frequency_ghz': ((548000 + frequencies) / 1000).tolist(),
        'checkpoint_selection': 'minimum validation loss; test evaluated only after training',
        'scope': 'New held-out measurements at known frequencies in the same available dataset; not unseen-frequency or external-session validation. Single fixed split; not cross-validation.',
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
    }
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'source').mkdir(exist_ok=True)
    for filename in ('experiment_independent.py', 'THz_data.py', 'model_final.py', 'test_independent_split.py'):
        shutil.copyfile(Path(__file__).resolve().parent / filename, out / 'source' / filename)
    print(f"Device: {device}; original train/val/test: {config['original_split_counts']}", flush=True)
    print(f"After train-only augmentation: {config['effective_split_counts']}; features: {mask.sum()}", flush=True)
    print(f'Cross-split identical spectra: {overlaps}', flush=True)

    torch.manual_seed(44)
    np.random.seed(44)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(44)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    datasets = {k: SpectralDataset(x[k], y[k]) for k in x}
    train_loader = DataLoader(datasets['train'], batch_size=64, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=64, shuffle=False)
    model = SpectralWavelengthPredictor(num_labels=len(frequencies), mode='classification').to(device)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=args.epochs,
        learning_rate=0.0015, device=device, save_path=str(out / 'checkpoints/best.pt'),
    )
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, args.epochs + 1), train_losses, label='Training loss')
    plt.plot(np.arange(1, args.epochs + 1), val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.title('Original-measurement holdout: training-only preprocessing and augmentation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / 'loss.png')
    plt.close()
    predictions, targets, _ = evaluate_and_save(
        datasets['test'], batch_size=64, output_dir=str(out), mode='classification',
        device=device, num_epochs=args.epochs, learning_rate=0.0015,
    )
    results = manifest.iloc[indices['test']].reset_index(drop=True).copy()
    assert np.array_equal(results.label_index.to_numpy(), targets)
    results['pred_label'] = predictions
    results['predicted_frequency_ghz'] = (548000 + frequencies[predictions]) / 1000
    errors_mhz = frequencies[predictions] - frequencies[targets]
    results['absolute_error_mhz'] = np.abs(errors_mhz)
    results['is_correct'] = (predictions == targets).astype(int)
    results.to_csv(out / 'prediction_results_frequency.csv', index=False)
    metrics_path = out / 'evaluation_metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update({
        'correct_samples': int(results.is_correct.sum()),
        'mae_mhz': float(np.abs(errors_mhz).mean()),
        'rmse_mhz': float(np.sqrt(np.mean(errors_mhz.astype(float) ** 2))),
        'max_error_mhz': int(np.abs(errors_mhz).max()),
        'best_epoch': int(np.argmin(val_losses) + 1),
        'best_val_loss': float(min(val_losses)),
        'protocol': config['protocol'],
    })
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    config['elapsed_seconds'] = time.monotonic() - started
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Completed in {config['elapsed_seconds']:.1f}s; results: {out}", flush=True)


if __name__ == '__main__':
    main()
