import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from scipy.fftpack import dct, idct
import collections
import pickle

#LEARNING_RATE = 0.2 
#(in case you want to change the learning rate, add it in: optimizer = optim.Adam(model.parameters()))

# Configuration
CONFIG = {
    'grid_size': 28,
    'num_modes': 100,
    'num_classes': 10,
    'valida_split': 0.2,
    'epochs': 100,
    'batch_size': 512,
    'activation_functions': ['relu'],
    'hidden_layers': [2, 2, 2],
    'dct_final_side_length': 10,
    'POR': 1  # Portion of total data to use (e.g., 0.1 for 10%)
}

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

def generate_nm_pairs(num_modes):
    side = int(np.ceil(np.sqrt(num_modes)))
    n, m = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1))
    return np.vstack([n.ravel(), m.ravel()]).T[:num_modes]

def generate_mode_shapes(grid_size, num_modes, Lx, Ly):
    start_time = time.time()
    print('Generating Mode Shapes...', flush=True)
    x = np.linspace(0, Lx, grid_size)
    y = np.linspace(0, Ly, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    pairs = generate_nm_pairs(num_modes)
    modes_2d = np.zeros((num_modes, grid_size, grid_size))
    for i, (m, n) in enumerate(pairs):
        modes_2d[i] = np.sin(m * np.pi * X_grid / Lx) * np.sin(n * np.pi * Y_grid / Ly)
    modes_flat = modes_2d.reshape(num_modes, -1)
    print(f"Mode shape generation completed in {time.time() - start_time:.2f} seconds.", flush=True)
    return modes_flat

def load_preprocess_mnist():
    start_time = time.time()
    print("Loading and preprocessing MNIST data...", flush=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.float() / 255.0
    y_train = train_dataset.targets
    x_test = test_dataset.data.float() / 255.0
    y_test = test_dataset.targets

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    y_train_one_hot = torch.eye(CONFIG['num_classes'])[y_train]
    y_test_one_hot = torch.eye(CONFIG['num_classes'])[y_test]

    por_train_samples = int(x_train_flat.shape[0] * CONFIG['POR'])
    por_test_samples = int(x_test_flat.shape[0] * CONFIG['POR'])

    train_indices = np.random.choice(x_train_flat.shape[0], por_train_samples, replace=False)
    test_indices = np.random.choice(x_test_flat.shape[0], por_test_samples, replace=False)

    x_train_flat = x_train_flat[train_indices]
    x_test_flat = x_test_flat[test_indices]
    y_train_one_hot = y_train_one_hot[train_indices]
    y_test_one_hot = y_test_one_hot[test_indices]
    y_train = y_train[train_indices]
    y_test = y_test[test_indices]

    print(f"Data loaded and sampled ({CONFIG['POR']*100:.0f}% of total) in {time.time() - start_time:.2f} seconds.", flush=True)
    print(f"Training samples: {x_train_flat.shape[0]}, Test samples: {x_test_flat.shape[0]}", flush=True)
    return x_train_flat.numpy(), x_test_flat.numpy(), y_train_one_hot.numpy(), y_test_one_hot.numpy(), y_train.numpy(), y_test.numpy()

def compute_features(x_train_flat, x_test_flat, modes_flat, num_modes):
    start_time = time.time()
    print(f"Calculating {num_modes} DRIFT features...", flush=True)
    x_train_drift = cosine_similarity(x_train_flat, modes_flat)
    x_test_drift = cosine_similarity(x_test_flat, modes_flat)
    print(f"DRIFT features in {time.time() - start_time:.2f} seconds.", flush=True)

    start_time = time.time()
    print("Calculating PCA features...", flush=True)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    pca = PCA(n_components=num_modes)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    print(f"PCA features in {time.time() - start_time:.2f} seconds.", flush=True)
    return x_train_drift, x_test_drift, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca

def compute_dct_features(x_train_flat, x_test_flat, final_side_length, original_grid_size):
    start_time = time.time()
    print(f"Calculating DCT features with final side length {final_side_length}...", flush=True)

    if final_side_length > original_grid_size or final_side_length <= 0:
        raise ValueError(
            f"dct_final_side_length ({final_side_length}) must be between 1 and original_grid_size ({original_grid_size})."
        )

    def apply_dct_reduction(images, target_side, grid_size):
        transformed_images = np.zeros((images.shape[0], target_side * target_side))
        for i, img_flat in enumerate(images):
            img_2d = img_flat.reshape(grid_size, grid_size)
            dct_2d = dct(dct(img_2d.T, norm='ortho').T, norm='ortho')
            reduced_dct = dct_2d[:target_side, :target_side]
            transformed_images[i] = reduced_dct.flatten()
        return transformed_images

    x_train_dct = apply_dct_reduction(x_train_flat, final_side_length, original_grid_size)
    x_test_dct = apply_dct_reduction(x_test_flat, final_side_length, original_grid_size)

    print(f"DCT features in {time.time() - start_time:.2f} seconds.", flush=True)
    return x_train_dct, x_test_dct

class SimpleNN(nn.Module):
    def __init__(self, input_shape, activation_name='relu'):
        super(SimpleNN, self).__init__()
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

        layers = []
        layers.append(nn.Linear(input_shape, CONFIG['hidden_layers'][0]))
        layers.append(self.activation)

        for i in range(1, len(CONFIG['hidden_layers'])):
            layers.append(nn.Linear(CONFIG['hidden_layers'][i - 1], CONFIG['hidden_layers'][i]))
            layers.append(self.activation)

        layers.append(nn.Linear(CONFIG['hidden_layers'][-1], CONFIG['num_classes']))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

def train_evaluate_model(model, x_train, x_test, y_train, y_test, y_test_labels, name, activation):
    start_time = time.time()
    print(f"\n--- Starting training for {name} ({activation}) ---", flush=True)
    print("-" * (len(name) + len(activation) + 30), flush=True)

    x_train_tensor = torch.from_numpy(x_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    x_test_tensor = torch.from_numpy(x_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    train_size = int((1 - CONFIG['valida_split']) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    print(f"{'Epoch':<5} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<9}", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, labels_idx = torch.max(labels.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels_idx).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_train / total_train
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                _, labels_idx = torch.max(labels.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels_idx).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_accuracy = correct_val / total_val
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_accuracy)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            print(f"{epoch + 1:<5} | {epoch_loss:<12.4f} | {epoch_accuracy:<11.4f} | {val_epoch_loss:<10.4f} | {val_epoch_accuracy:<9.4f}", flush=True)

    print("-" * 60, flush=True)
    print(f"Finished training for {name} ({activation}) in {time.time() - start_time:.2f} seconds.", flush=True)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    y_pred_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            _, labels_idx = torch.max(labels.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels_idx).sum().item()

    accuracy = correct_test / total_test
    y_pred_labels = np.array(y_pred_list)
    top1_accuracy = accuracy_score(y_test_labels, y_pred_labels)

    return history, accuracy, top1_accuracy

def main():
    modes_flat = generate_mode_shapes(
        CONFIG['grid_size'], CONFIG['num_modes'], CONFIG['grid_size'], CONFIG['grid_size']
    )
    x_train_flat, x_test_flat, y_train_one_hot, y_test_one_hot, y_train, y_test = load_preprocess_mnist()
    
    x_train_drift, x_test_drift, x_train_scaled, x_test_scaled, x_train_pca, x_test_pca = compute_features(
        x_train_flat, x_test_flat, modes_flat, CONFIG['num_modes']
    )
    x_train_dct, x_test_dct = compute_dct_features(
        x_train_flat, x_test_flat, CONFIG['dct_final_side_length'], CONFIG['grid_size']
    )

    feature_set_info = [
        (x_train_drift, x_test_drift, "DRIFT"),
        (x_train_pca, x_test_pca, "PCA"),
        (x_train_scaled, x_test_scaled, "RAW Data"),
        (x_train_dct, x_test_dct, "DCT")
    ]

    all_methods_epoch_histories = collections.defaultdict(list)
    final_results = []

    print("\n" + "="*80, flush=True)
    print("Beginning Training of All Models", flush=True)
    print("="*80 + "\n", flush=True)

    for activation in CONFIG['activation_functions']:
        for x_train, x_test, name in feature_set_info:
            model = SimpleNN(x_train.shape[1], activation_name=activation).to(device)
            history, accuracy, top1_accuracy = train_evaluate_model(
                model, x_train, x_test, y_train_one_hot, y_test_one_hot, y_test, name, activation
            )
            all_methods_epoch_histories[name].append(history)
            final_results.append({'name': name, 'activation': activation, 'accuracy': accuracy, 'top1': top1_accuracy})

    print("\n" + "="*80, flush=True)
    print("--- Detailed Epoch Progress (Synchronized View After All Training) ---", flush=True)
    print("="*80 + "\n", flush=True)
    header = f"{'Epoch':<5}"
    for method_name in all_methods_epoch_histories.keys():
        header += f" | {method_name} (Trn L / Val L / Trn A / Val A)"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    num_epochs = CONFIG['epochs']
    for epoch in range(num_epochs):
        line_output = f"{epoch + 1:<5}"
        for method_name in all_methods_epoch_histories.keys():
            history = all_methods_epoch_histories[method_name][0]
            train_loss = history['loss'][epoch]
            val_loss = history['val_loss'][epoch]
            train_acc = history['accuracy'][epoch]
            val_acc = history['val_accuracy'][epoch]
            line_output += f" | {train_loss:.4f} / {val_loss:.4f} / {train_acc:.4f} / {val_acc:.4f}"
        print(line_output, flush=True)

    print("\n--- Final Test Accuracies ---", flush=True)
    for res in final_results:
        print(f"{res['name']} ({res['activation']}) - Test accuracy: {res['accuracy']:.4f}, Top-1: {res['top1']:.4f}", flush=True)

    # Save histories and CONFIG for plotting
    with open('training_histories10_100_100.pkl', 'wb') as f:
        pickle.dump({'histories': all_methods_epoch_histories, 'CONFIG': CONFIG}, f)

    return all_methods_epoch_histories

if __name__ == "__main__":
    histories = main()
