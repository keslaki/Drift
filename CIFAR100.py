import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from scipy.fftpack import dct
from torch.utils.data import DataLoader, TensorDataset

# --- Hyperparameters ---
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS =400
HIDDEN_LAYERS = [64, 128, 64]
FFT_SIZE = 5
NUM_MODES = 25
GRID_SIZE = 32
PADDING = 5  # Dynamic padding size for DRIFT
DRIFT_GRID_SIZE = GRID_SIZE + 2 * PADDING  # New grid size for DRIFT (42x42)

# Set a fixed random seed for all models
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate mode shapes
def generate_nm_pairs(num_modes):
    side = int(np.ceil(np.sqrt(num_modes)))
    n, m = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1))
    return np.vstack([n.ravel(), m.ravel()]).T[:num_modes]

def generate_mode_shapes(grid_size, num_modes, Lx, Ly):
    x = np.linspace(0, Lx, grid_size)
    y = np.linspace(0, Ly, grid_size)
    X, Y = np.meshgrid(x, y)
    pairs = generate_nm_pairs(num_modes)
    modes = np.zeros((num_modes, grid_size, grid_size))
    for i, (m, n) in enumerate(pairs):
        modes[i] = np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
    return modes.reshape(num_modes, -1)

# Load CIFAR-100 data with padding for DRIFT
def load_preprocess_cifar100(padding=PADDING):
    print("Loading CIFAR-100 data...")
    start = time.time()
    train_dataset = datasets.CIFAR100(root='./', train=True, download=True)
    test_dataset = datasets.CIFAR100(root='./', train=False, download=True)
    x_train = train_dataset.data
    y_train = np.array(train_dataset.targets)
    x_test = test_dataset.data
    y_test = np.array(test_dataset.targets)
    
    # Normalize original data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    # Create padded data for DRIFT
    x_train_padded = np.zeros((x_train.shape[0], DRIFT_GRID_SIZE, DRIFT_GRID_SIZE, 3))
    x_test_padded = np.zeros((x_test.shape[0], DRIFT_GRID_SIZE, DRIFT_GRID_SIZE, 3))
    for i in range(x_train.shape[0]):
        for c in range(3):  # R, G, B channels
            x_train_padded[i, padding:-padding, padding:-padding, c] = x_train[i, :, :, c]
    for i in range(x_test.shape[0]):
        for c in range(3):
            x_test_padded[i, padding:-padding, padding:-padding, c] = x_test[i, :, :, c]
    
    elapsed = time.time() - start
    print(f"CIFAR-100 data loaded in {elapsed:.2f} seconds")
    return x_train, x_test, x_train_padded, x_test_padded, y_train, y_test, elapsed

# Compute drift features
def compute_drift_features(x_train, x_test, modes_flat, num_modes):
    print(f"Calculating {num_modes} DRIFT features per channel...")
    start = time.time()
    x_train_drift = []
    x_test_drift = []
    for channel in range(3):  # Process R, G, B channels
        x_train_flat = x_train[:, :, :, channel].reshape(-1, DRIFT_GRID_SIZE * DRIFT_GRID_SIZE)
        x_test_flat = x_test[:, :, :, channel].reshape(-1, DRIFT_GRID_SIZE * DRIFT_GRID_SIZE)
        train_drift = np.dot(x_train_flat, modes_flat.T)
        test_drift = np.dot(x_test_flat, modes_flat.T)
        x_train_drift.append(train_drift)
        x_test_drift.append(test_drift)
    x_train_drift = np.concatenate(x_train_drift, axis=1)
    x_test_drift = np.concatenate(x_test_drift, axis=1)
    elapsed = time.time() - start
    print(f"Drift features computed in {elapsed:.2f} seconds")
    return x_train_drift, x_test_drift, elapsed

# Compute PCA features
def compute_pca_features(x_train, x_test, num_modes):
    print("Calculating PCA features per channel...")
    start = time.time()
    x_train_pca = []
    x_test_pca = []
    scaler = StandardScaler()
    for channel in range(3):  # Process R, G, B channels
        x_train_flat = x_train[:, :, :, channel].reshape(-1, GRID_SIZE * GRID_SIZE)
        x_test_flat = x_test[:, :, :, channel].reshape(-1, GRID_SIZE * GRID_SIZE)
        x_train_scaled = scaler.fit_transform(x_train_flat)
        x_test_scaled = scaler.transform(x_test_flat)
        pca = PCA(n_components=num_modes)
        train_pca = pca.fit_transform(x_train_scaled)
        test_pca = pca.transform(x_test_scaled)
        x_train_pca.append(train_pca)
        x_test_pca.append(test_pca)
    x_train_pca = np.concatenate(x_train_pca, axis=1)
    x_test_pca = np.concatenate(x_test_pca, axis=1)
    elapsed = time.time() - start
    print(f"PCA features computed in {elapsed:.2f} seconds")
    return x_train_pca, x_test_pca, elapsed

# DCT feature extraction
def extract_dct_features(x, size=5):
    print(f"Extracting DCT features (size={size}) per channel...")
    start = time.time()
    feats_train = []
    feats_test = []
    for channel in range(3):  # Process R, G, B channels
        feats = []
        for img in x[:, :, :, channel]:
            dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
            feats.append(dct_img[:size, :size].flatten())
        feats_train.append(np.array(feats))
    feats_test = []
    for channel in range(3):
        feats = []
        for img in x_test[:, :, :, channel]:
            dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
            feats.append(dct_img[:size, :size].flatten())
        feats_test.append(np.array(feats))
    elapsed = time.time() - start
    print(f"DCT features computed in {elapsed:.2f} seconds")
    return np.concatenate(feats_train, axis=1), np.concatenate(feats_test, axis=1), elapsed

# Generate mode shapes for DRIFT
modes_flat = generate_mode_shapes(DRIFT_GRID_SIZE, NUM_MODES, DRIFT_GRID_SIZE, DRIFT_GRID_SIZE)

# Load data
x_train, x_test, x_train_drift, x_test_drift, y_train_labels, y_test_labels, data_load_time = load_preprocess_cifar100()

# Compute features and collect prep times
prep_times = {'Data Loading': data_load_time}
x_train_drift, x_test_drift, prep_times['Drift'] = compute_drift_features(x_train_drift, x_test_drift, modes_flat, NUM_MODES)
x_train_pca, x_test_pca, prep_times['PCA'] = compute_pca_features(x_train, x_test, NUM_MODES)
dct_x_train, dct_x_test, prep_times['DCT'] = extract_dct_features(x_train, size=FFT_SIZE)
prep_times['Raw'] = 0.0

# Prepare feature sets
feature_sets = [
    (dct_x_train, dct_x_test, "DCT"),
    (x_train_pca, x_test_pca, "PCA"),
    (x_train.reshape(-1, 3 * GRID_SIZE * GRID_SIZE), x_test.reshape(-1, 3 * GRID_SIZE * GRID_SIZE), "Raw"),
    (x_train_drift, x_test_drift, "Drift")
]

# Neural network class
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 100))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Store histories and training times
histories = []
train_times = {}

for feat_train, feat_test, name in feature_sets:
    print(f"\nTraining on {name}...")
    start = time.time()
    model = NeuralNet(feat_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_tensor = torch.tensor(feat_train, dtype=torch.float32, device=device)
    test_tensor = torch.tensor(feat_test, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long, device=device)
    y_test_tensor = torch.tensor(y_test_labels, dtype=torch.long, device=device)

    train_dataset = TensorDataset(train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).float().mean().item()

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_outputs = model(test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_test_tensor).float().mean().item()

        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss.item())

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    train_times[name] = time.time() - start
    print(f"Training on {name} completed in {train_times[name]:.2f} seconds")
    histories.append({'name': name, 'history': history})

# Print times
print("\nFeature Preparation and Training Times:")
for name in ['Data Loading'] + [fs[2] for fs in feature_sets]:
    if name == 'Data Loading':
        print(f"{name}: {prep_times[name]:.2f}s")
    else:
        print(f"{name} - Prep Time: {prep_times[name]:.2f}s, Training Time: {train_times[name]:.2f}s")
