import numpy as np
from pathlib import Path

def load_file_to_array(file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a file into a tuple of arrays.
    spectra_wavelength: shape (spectrum_length,)
    spectra_data: shape (num_samples, spectrum_length)
    """
    data = np.loadtxt(file)
    spectra_data = data[:, 1]
    spectra_wavelength = data[:, 0]
    return spectra_data, spectra_wavelength


def load_all_data(return_paths=False):

    labels = []
    data = []
    paths = []
    root = Path(__file__).resolve().parent
    legacy_dir = root / 'data_all' / 'data'
    data_dir = legacy_dir if legacy_dir.is_dir() else root / 'data_1mhz'
    for label in np.arange(591, 811):
        folder = data_dir / (f'Thz548_{label}' if data_dir == legacy_dir else f'with thz 548-{label}')
        files = sorted(folder.glob(f'chafen-with thz 548-{label}_*.txt'))
        if len(files) != 5:
            raise ValueError(f'Expected 5 spectra for label {label}, found {len(files)} in {folder}')
        for file in files:
            spectra_data, _ = load_file_to_array(file)
            labels.append(label)
            data.append(spectra_data)
            paths.append(file)

    labels = np.array(labels)
    data = np.array(data)
    if not np.isfinite(data).all():
        raise ValueError('Spectra contain NaN or infinite values')

    return (labels, data, paths) if return_paths else (labels, data)

## Dimensionality reduction
def data_reduction(data: np.ndarray, reduction_limit: float = 3.0) -> np.ndarray:
    """
    data: shape (num_samples, spectrum_length)
    
    return: shape (num_samples, num_features)
    data has many zero points, we need to remove them
    """
    all_spectra = abs(np.sum(data, axis=0))  # shape (spectrum_length,), sum over samples

    index = all_spectra > all_spectra.mean()/reduction_limit
    data_reduced = data[:, index]

    all_spectra = abs(np.sum(data_reduced, axis=0))  # shape (num_features,), num_features = spectrum intensity larger than mean/3

    print(all_spectra.min(), all_spectra.max(), all_spectra.shape)

    return data_reduced





if __name__ == "__main__":
    labels, data = load_all_data()
    print(labels.shape, data.shape)
