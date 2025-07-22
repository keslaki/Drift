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
EPOCHS = 50
HIDDEN_LAYERS = [128, 256, 128]
FFT_SIZE = 7  # Not used now, but kept in case needed
NUM_MODES = 49
GRID_SIZE = 28
DROPOUT = 0.2  # Dropout rate; set to 0.0 to disable dropout

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

# Load FashionMNIST data
def load_preprocess_fashion_mnist():
    print("Loading FashionMNIST data...")
    start = time.time()
    train_dataset = datasets.FashionMNIST(root='./', train=True, download=True)
    test_dataset = datasets.FashionMNIST(root='./', train=False, download=True)
    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    elapsed = time.time() - start
    print(f"FashionMNIST data loaded in {elapsed:.2f} seconds")
    return x_train, x_test, y_train, y_test, elapsed

# Compute drift features
def compute_drift_features(x_train_flat, x_test_flat, modes_flat, num_modes):
    print(f"Calculating {num_modes} DRIFT features...")
    start = time.time()
    x_train_drift = np.dot(x_train_flat, modes_flat.T)
    x_test_drift = np.dot(x_test_flat, modes_flat.T)
    elapsed = time.time() - start
    print(f"Drift features computed in {elapsed:.2f} seconds")
    return x_train_drift, x_test_drift, elapsed

# Compute PCA features
def compute_pca_features(x_train_flat, x_test_flat, num_modes):
    print("Calculating PCA features...")
    start = time.time()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    pca = PCA(n_components=num_modes)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    elapsed = time.time() - start
    print(f"PCA features computed in {elapsed:.2f} seconds")
    return x_train_pca, x_test_pca, elapsed

# DCT feature extraction
def extract_dct_features(x, size=10):
    print(f"Extracting DCT features (size={size})...")
    start = time.time()
    feats = []
    for img in x:
        dct_img = dct(dct(img.T, norm='ortho').T, norm='ortho')
        feats.append(dct_img[:size, :size].flatten())
    elapsed = time.time() - start
    print(f"DCT features computed in {elapsed:.2f} seconds")
    return np.array(feats), elapsed

# Generate mode shapes
modes_flat = generate_mode_shapes(GRID_SIZE, NUM_MODES, GRID_SIZE, GRID_SIZE)

# Load data
x_train, x_test, y_train_labels, y_test_labels, data_load_time = load_preprocess_fashion_mnist()

# Flatten images for drift & PCA
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# Compute features and collect prep times
prep_times = {'Data Loading': data_load_time}
x_train_drift, x_test_drift, prep_times['Drift'] = compute_drift_features(x_train_flat, x_test_flat, modes_flat, NUM_MODES)
x_train_pca, x_test_pca, prep_times['PCA'] = compute_pca_features(x_train_flat, x_test_flat, NUM_MODES)
dct_x_train, prep_times['DCT'] = extract_dct_features(x_train, size=FFT_SIZE)
dct_x_test, _ = extract_dct_features(x_test, size=FFT_SIZE)
prep_times['Raw'] = 0.0  # No extra feature extraction for Raw

# Prepare feature sets (exclude FFT)
feature_sets = [
    (dct_x_train, dct_x_test, "DCT"),
    (x_train_pca, x_test_pca, "PCA"),
    (x_train.reshape(-1, 28*28), x_test.reshape(-1, 28*28), "Raw"),
    (x_train_drift, x_test_drift, "Drift")
]

# Neural network class with dropout
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if DROPOUT > 0.0:
                layers.append(nn.Dropout(DROPOUT))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 10))
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

        # Save history
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
