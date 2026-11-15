import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import os

# Custom dataset class
class SpectralDataset(Dataset):
    def __init__(self, spectra, labels, transform=None):
        """
        spectra: numpy array of shape (n_samples, 32000)
        labels: numpy array of shape (n_samples,) - laser wavelength values
        """
        self.spectra = torch.FloatTensor(spectra)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        wavelength = self.labels[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
            
        return spectrum, wavelength

# Neural network model
class SpectralWavelengthPredictor(nn.Module):
    def __init__(self, num_labels=300, mode='regression'):
        super(SpectralWavelengthPredictor, self).__init__()
        self.mode = mode
        
        # 1D convolutions for feature extraction and dimensionality reduction
        self.conv_blocks = nn.Sequential(
            # First convolutional block
            nn.Conv1d(1, 32, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3),
            
            # Fourth convolutional block
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),  # Adaptive pooling to a fixed length
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer
        if mode == 'regression':
            self.output_layer = nn.Linear(128, 1)  # Regression: predict continuous wavelength values
        else:
            self.output_layer = nn.Linear(128, num_labels)  # Classification: predict wavelength classes
        
    def forward(self, x):
        # Reshape the input (batch_size, 32000) -> (batch_size, 1, 32000)
        x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        for layer in self.conv_blocks:
            if x.device.type == 'mps' and isinstance(layer, nn.AdaptiveAvgPool1d):
                # MPS does not support non-divisible adaptive pooling sizes.
                x = layer(x.cpu()).to(x.device)
            else:
                x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Output
        x = self.output_layer(x)
        
        if self.mode == 'regression':
            x = x.squeeze()
            
        return x

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, 
                learning_rate=0.001, device='cuda', save_path='checkpoints/best.pt'):
    
    criterion = nn.MSELoss() if model.mode == 'regression' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for spectra, labels in train_loader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectra)
            
            if model.mode == 'regression':
                loss = criterion(outputs, labels)
            else:
                # Classification requires integer class indices
                loss = criterion(outputs, labels.long())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(device)
                labels = labels.to(device)
                
                outputs = model(spectra)
                
                if model.mode == 'regression':
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels.long())
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    # Save per-epoch losses to a CSV file
    loss_data = {
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    loss_df = pd.DataFrame(loss_data)
    loss_path = os.path.join(os.path.dirname(os.path.dirname(save_path)), 'training_losses.csv')
    loss_df.to_csv(loss_path, index=False)
    print(f"Training losses saved to '{loss_path}'")
    
    return train_losses, val_losses


# Evaluation function
def evaluate_model(model, test_loader, mode='regression', device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate model performance (classification: accuracy/F1/confusion matrix; regression: MAE/RMSE/R²)
    """
    model = model.to(device)
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for spectra, labels in tqdm(test_loader, desc='Evaluating'):
            spectra = spectra.to(device)
            if mode == 'classification':
                logits = model(spectra)
                batch_preds = logits.argmax(dim=1).cpu().numpy()
                predictions.extend(batch_preds)
                targets.extend(labels.cpu().numpy().astype(int))
            else:
                outputs = model(spectra)
                batch_preds = outputs.view(-1).cpu().numpy()
                predictions.extend(batch_preds)
                targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    if mode == 'classification':
        acc = float(accuracy_score(targets, predictions))
        f1 = float(f1_score(targets, predictions, average='macro'))
        cm = confusion_matrix(targets, predictions)
        print(f'Accuracy: {acc:.4f}')
        print(f'F1-macro: {f1:.4f}')
        return predictions, targets, {'accuracy': acc, 'f1_macro': f1, 'confusion_matrix': cm.tolist()}
    else:
        mae = float(np.mean(np.abs(predictions - targets)))
        rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
        r2 = float(1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2))
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'R²: {r2:.4f}')
        return predictions, targets, {'mae': mae, 'rmse': rmse, 'r2': r2}

def evaluate_and_save(test_dataset,
                      batch_size,
                      output_dir,
                      mode='regression',
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      num_epochs=100,
                      learning_rate=0.001):
    """
    Load the best model weights, evaluate the supplied test set, and save:
    - Predictions to `prediction_results.csv`
    - Evaluation metrics to `evaluation_metrics.json`
    - Visualizations to `prediction_results.png`

    Return (predictions, targets, metrics)
    """

    # Ensure the directories exist
    os.makedirs(output_dir or '.', exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the best model
    model = SpectralWavelengthPredictor(
        num_labels=len(np.unique(test_dataset.labels)),
        mode=mode
    ).to(device)

    best_path = os.path.join(output_dir, 'checkpoints', 'best.pt')
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    print(f"Load model: {best_path}")

    # Evaluate
    predictions, targets, metrics = evaluate_model(model, test_loader, mode=mode, device=device)

    # Save predictions
    if mode == 'classification':
        results_data = {
            'true_label': targets,
            'pred_label': predictions,
            'is_correct': (predictions == targets).astype(int)
        }
    else:
        results_data = {
            'true_values': targets,
            'predicted_values': predictions,
            'absolute_error': np.abs(predictions - targets)
        }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    print("Predictions saved to 'prediction_results.csv'")

    # Save evaluation metrics
    if mode == 'classification':
        metrics_data = {
            'accuracy': float(metrics['accuracy']),
            'f1_macro': float(metrics['f1_macro']),
            'mode': mode,
            'num_samples': len(targets),
            'training_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    else:
        metrics_data = {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'mode': mode,
            'num_samples': len(targets),
            'training_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Evaluation metrics saved to 'evaluation_metrics.json'")

    # Plot prediction results
    if mode == 'classification':
        cm = np.array(metrics['confusion_matrix'])
        n_classes = cm.shape[0]
        fig_w = min(30, max(10, n_classes / 8))
        fig_h = min(30, max(8, n_classes / 10))
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
        plt.title(f'Confusion Matrix (n={n_classes})')
        plt.colorbar()
        if n_classes <= 50:
            step = 1
        elif n_classes <= 100:
            step = 2
        elif n_classes <= 200:
            step = 5
        else:
            step = 10
        tick_positions = np.arange(0, n_classes, step)
        plt.xticks(tick_positions, tick_positions, rotation=45, fontsize=6)
        plt.yticks(tick_positions, tick_positions, fontsize=6)
        plt.tick_params(axis='both', which='both', length=2)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        plt.xlabel('True THz')
        plt.ylabel('Predicted THz')
        plt.title(f'Prediction Results (R² = {metrics["r2"]:.4f}), mode = {mode}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'prediction_results.png'))
        plt.close()

    return predictions, targets, metrics

def train_val_test_split_by_label(spectra_data, labels, val_count=1, test_count=1, seed=42):
    """
    For each label, randomly select val_count validation samples and test_count test samples; use the rest for training.
    Return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng = np.random.default_rng(seed)
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        val_idx = idx[:val_count]
        test_idx = idx[val_count:val_count + test_count]
        train_idx = idx[val_count + test_count:]

        X_val.extend(spectra_data[val_idx])
        y_val.extend(labels[val_idx])

        X_test.extend(spectra_data[test_idx])
        y_test.extend(labels[test_idx])

        X_train.extend(spectra_data[train_idx])
        y_train.extend(labels[train_idx])

    return (np.array(X_train), np.array(X_val), np.array(X_test),
            np.array(y_train), np.array(y_val), np.array(y_test))

# Data processing and training
def prepare_and_train(spectra_data, labels_data, debug=False, augment=False, mode='regression', num_epochs=100, learning_rate=0.001, batch_size=32, random_state=42, output_dir='.', device=None):
    """
    spectra_data: numpy array of shape (n_samples, 32000)
    labels_data: numpy array of shape (n_samples,) 
    """
    # Split training, validation, and test samples by label
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_by_label(
        spectra_data, labels_data, val_count=1, test_count=1, seed=random_state
    )

    # Fit the scaler on training data only to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Augment the training set only
    if augment and isinstance(augment, int):
        X_train, y_train = offline_augment(X_train, y_train, copies=augment, noise_std=0.02)
    
    # Create data loaders
    train_dataset = SpectralDataset(X_train, y_train)
    val_dataset = SpectralDataset(X_val, y_val)
    test_dataset = SpectralDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    if debug:
        device = 'cpu'
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Training device: {device}; train/val/test samples: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}', flush=True)
    model = SpectralWavelengthPredictor(
        num_labels=len(np.unique(labels_data)),
        mode=mode
    ).to(device)
    
    # Ensure the output directory exists
    os.makedirs(output_dir or '.', exist_ok=True)

    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs, learning_rate=learning_rate, device=device,
        save_path=os.path.join(output_dir, 'checkpoints', 'best.pt')
    )
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss, mode = {mode}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # Use the separate evaluation function
    _ = evaluate_and_save(
        test_dataset=test_dataset,
        batch_size=batch_size,
        output_dir=output_dir,
        mode=mode,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    return model, scaler

def offline_augment(x, y, copies=10,
                    noise_std=0.01,          # Noise = noise_std × each spectrum's own RMS
                    seed=None,
                   ):
    """
    x : numpy.ndarray, shape (N, 32000)
    y : numpy.ndarray, shape (N,)
    return : (x_aug, y_aug)  shape ((N*(copies+1), 32000), (N*(copies+1),))
             Includes the originals plus copies noisy versions
    """
    N = x.shape[0]
    specs   = [x]          # Originals
    targets = [y]

    rng = np.random.default_rng(seed)

    for _ in range(copies):
        # 1 Scale
        scale = rng.uniform(1.0-noise_std, 1.0+noise_std,  size=(N, 1))
        x_scaled = x * scale

        # 2 Gaussian noise (scaled by each spectrum's RMS)
        rms   = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        noise = rng.normal(0, rms*noise_std, size=x.shape)
        x_noisy = x_scaled + noise


        # 3 Random offsets
        shift = rng.uniform(-rms*noise_std, rms*noise_std, size=(N, 1))
        x_shifted = x_noisy + shift

        specs.append(x_shifted.astype(np.float32))
        targets.append(y)

    x_aug = np.concatenate(specs,   axis=0)
    y_aug = np.concatenate(targets, axis=0)
    return x_aug, y_aug

if __name__ == "__main__":

    import argparse
    import joblib
    import platform
    import time
    from THz_data import load_all_data, data_reduction
    parser = argparse.ArgumentParser(description='Rerun the original 220-class spectral model')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.monotonic()
    # spectra, label_wavelength = load_data(['data{}.txt'.format(i) for i in range(1, 3)])
    labels, spectra = load_all_data()
    all_spectra = abs(np.sum(spectra, axis=0))
    feature_mask = all_spectra > all_spectra.mean() / 20
    np.save(os.path.join(args.output_dir, 'feature_mask.npy'), feature_mask)
    spectra = data_reduction(spectra, 20)
    labels = labels.astype(int)
    raw_labels = np.unique(labels)

    n_samples = spectra.shape[0]
    n_features = spectra.shape[1]
    labels = labels - labels.min()

    n_labels = len(np.unique(labels))
    print(spectra.shape, labels.shape, n_labels)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = {
        'mode': 'classification', 'epochs': args.epochs, 'learning_rate': 0.0015,
        'batch_size': 64, 'device': device, 'raw_samples': n_samples,
        'raw_features': len(feature_mask), 'retained_features': n_features,
        'num_classes': n_labels, 'frequency_ghz': (548 + raw_labels / 1000).tolist(),
        'torch_numpy_seed': 44, 'split_seed': 42,
        'augmentation_seed': None,
        'augmentation_before_split': {'copies': 1, 'noise_std': 0.1},
        'augmentation_train_only': {'copies': 1, 'noise_std': 0.02},
        'evaluation_note': 'Original protocol retained: augmentation and feature selection precede splitting; test samples are not fully independent of training. Augmentation uses unseeded default_rng, as in the original script.',
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
    }
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    # Set random seeds for reproducibility
    torch.manual_seed(44)
    np.random.seed(44)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(44)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Ensure the checkpoint directory exists
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    spectra, labels = offline_augment(spectra, labels, copies=1, noise_std=0.1)

    model, scaler = prepare_and_train(spectra, labels, augment=1, debug=False, mode='classification', 
                                      num_epochs=args.epochs, learning_rate=0.0015, batch_size=64,
                                      output_dir=args.output_dir, device=device)
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    results = pd.read_csv(os.path.join(args.output_dir, 'prediction_results.csv'))
    results['true_frequency_ghz'] = 548 + raw_labels[results['true_label'].to_numpy()] / 1000
    results['predicted_frequency_ghz'] = 548 + raw_labels[results['pred_label'].to_numpy()] / 1000
    results['absolute_error_mhz'] = np.abs(results['pred_label'] - results['true_label'])
    results.to_csv(os.path.join(args.output_dir, 'prediction_results_frequency.csv'), index=False)
    config['elapsed_seconds'] = time.monotonic() - started
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Completed in {config['elapsed_seconds']:.1f} seconds. Results: {args.output_dir}", flush=True)
