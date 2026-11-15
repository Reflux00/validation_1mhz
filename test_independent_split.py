"""Held-out data must not affect fitted preprocessing or augmented training data."""
import numpy as np
from experiment_independent import prepare_independent_data


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(20, 100))
    labels = np.repeat(np.arange(4), 5)
    x, y, indices, mask, scaler = prepare_independent_data(spectra, labels)
    assert {k: len(v) for k, v in indices.items()} == {'train': 12, 'val': 4, 'test': 4}
    assert len(x['train']) == 48 and len(x['val']) == len(x['test']) == 4
    assert set(indices['train']).isdisjoint(indices['val'])
    assert set(indices['train']).isdisjoint(indices['test'])
    assert set(indices['val']).isdisjoint(indices['test'])
    changed = spectra.copy()
    changed[np.r_[indices['val'], indices['test']]] += 1e6
    x2, y2, indices2, mask2, scaler2 = prepare_independent_data(changed, labels)
    np.testing.assert_array_equal(mask, mask2)
    np.testing.assert_array_equal(scaler.mean_, scaler2.mean_)
    np.testing.assert_array_equal(scaler.scale_, scaler2.scale_)
    np.testing.assert_array_equal(x['train'], x2['train'])
    np.testing.assert_array_equal(y['train'], y2['train'])
    np.testing.assert_allclose(scaler.mean_, spectra[indices['train']][:, mask].mean(axis=0))
    assert scaler.n_samples_seen_ == 12
    for name in ('val', 'test'):
        np.testing.assert_array_equal(indices[name], indices2[name])
        np.testing.assert_array_equal(y[name], labels[indices[name]])
        np.testing.assert_allclose(x[name], scaler.transform(spectra[indices[name]][:, mask]))
        assert not np.array_equal(x[name], x2[name])
    print('PASS: disjoint originals; held-out values cannot affect preprocessing or train augmentation')
