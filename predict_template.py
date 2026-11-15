"""Classify raw two-column spectrum TXT files without parsing their filenames."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'runs/20261003_optimization/final_template/model.joblib')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = joblib.load(args.model)
    reference_axis = np.load(args.model.with_name('spectral_axis.npy'))
    # Original models use integer MHz labels; fine-grid models declare their tick unit.
    metadata_path = args.model.with_name('model_metadata.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    label_unit_mhz = float(metadata.get('label_unit_mhz', 1.0))
    if not np.isfinite(label_unit_mhz) or label_unit_mhz <= 0:
        raise ValueError('model_metadata.json: label_unit_mhz must be finite and positive')
    spectra = []
    for path in args.files:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape != (model.n_features_in_, 2) or not np.isfinite(data).all():
            raise ValueError(f'{path}: expected {model.n_features_in_} rows, 2 finite columns')
        if not np.allclose(data[:, 0], reference_axis, rtol=0, atol=1e-9):
            raise ValueError(f'{path}: spectral coordinates do not match the training grid')
        spectra.append(data[:, 1])
    pred = model.predict(np.asarray(spectra))
    results = pd.DataFrame({'file': [str(p) for p in args.files], 'predicted_frequency_ghz': (548000 + pred * label_unit_mhz) / 1000})
    print(results.to_string(index=False))
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
