import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fftpack import dct, idct

GRID_SIZE = 28
POR = 1.0  # Use full dataset
USE_SAMPLE_NORM = True  # True: use ||A|| from sample; False: norm=1

NUM_MODES_LIST = [16, 36, 49, 100]
DCT_FINAL_SIDES = [4, 6, 7, 10]

def load_mnist_all(por=1.0):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    all_data = torch.cat((train_dataset.data, test_dataset.data), dim=0).float() / 255.0
    n_samples = int(len(all_data) * por)
    indices = np.random.choice(len(all_data), n_samples, replace=False)
    all_data = all_data[indices].reshape(n_samples, -1).numpy()
    return all_data

def generate_mode_shapes(grid_size, num_modes, Lx=28, Ly=28):
    x = np.linspace(0, Lx, grid_size)
    y = np.linspace(0, Ly, grid_size)
    X, Y = np.meshgrid(x, y)
    side = int(np.ceil(np.sqrt(num_modes)))
    pairs = [(m,n) for m in range(1, side+1) for n in range(1, side+1)]
    pairs = pairs[:num_modes]

    modes = []
    for m, n in pairs:
        mode = np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
        modes.append(mode.flatten())
    return np.array(modes)

def compute_drift_features(samples, modes):
    return cosine_similarity(samples, modes)

def compute_pca_features(x_data, n_components):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_scaled)
    return x_pca, pca, scaler

def compute_dct_features(x, final_side, grid_size):
    n_samples = x.shape[0]
    dct_features = np.zeros((n_samples, final_side*final_side))
    for i in range(n_samples):
        img = x[i].reshape(grid_size, grid_size)
        img_dct = dct(dct(img.T, norm='ortho').T, norm='ortho')
        dct_features[i] = img_dct[:final_side, :final_side].flatten()
    return dct_features

def reconstruct_from_drift(sample_drift, modes, sample_norm=1.0):
    modes_norms = np.linalg.norm(modes, axis=1)
    modes_norms_sq = modes_norms**2
    weights = sample_drift * sample_norm * modes_norms
    projection_coeffs = weights / modes_norms_sq
    reconstructed = projection_coeffs @ modes
    return reconstructed

def reconstruct_from_pca(sample_pca, pca, scaler):
    x_scaled_recon = pca.inverse_transform(sample_pca.reshape(1, -1))
    x_recon = scaler.inverse_transform(x_scaled_recon)
    return x_recon.flatten()

def reconstruct_from_dct(sample_dct, final_side, grid_size):
    dct_2d = np.zeros((grid_size, grid_size))
    dct_2d[:final_side, :final_side] = sample_dct.reshape(final_side, final_side)
    img_recon = idct(idct(dct_2d.T, norm='ortho').T, norm='ortho')
    return img_recon.flatten()

def mse(x_true, x_pred):
    return np.mean((x_true - x_pred)**2)

def run_all_reconstructions():
    print("Loading ALL MNIST data (train + test combined)...")
    all_data = load_mnist_all(POR)
    print(f"Total samples: {all_data.shape[0]}")

    results = {'NUM_MODES': [], 'DRIFT_MEAN': [], 'DRIFT_STD': [],
               'PCA_MEAN': [], 'PCA_STD': [], 'DCT_MEAN': [], 'DCT_STD': []}

    for NUM_MODES, DCT_FINAL_SIDE in zip(NUM_MODES_LIST, DCT_FINAL_SIDES):
        print(f"\nProcessing NUM_MODES={NUM_MODES}, DCT_FINAL_SIDE={DCT_FINAL_SIDE}...")

        modes = generate_mode_shapes(GRID_SIZE, NUM_MODES)

        x_drift = compute_drift_features(all_data, modes)
        x_pca, pca, scaler = compute_pca_features(all_data, NUM_MODES)
        x_dct = compute_dct_features(all_data, DCT_FINAL_SIDE, GRID_SIZE)

        drift_errors = []
        pca_errors = []
        dct_errors = []

        for i in range(all_data.shape[0]):
            orig = all_data[i]
            norm_orig = np.linalg.norm(orig)
            sample_norm = norm_orig if USE_SAMPLE_NORM else 1.0

            drift_recon = reconstruct_from_drift(x_drift[i], modes, sample_norm=sample_norm)
            pca_recon = reconstruct_from_pca(x_pca[i], pca, scaler)
            dct_recon = reconstruct_from_dct(x_dct[i], DCT_FINAL_SIDE, GRID_SIZE)

            drift_errors.append(mse(orig, drift_recon))
            pca_errors.append(mse(orig, pca_recon))
            dct_errors.append(mse(orig, dct_recon))

            if (i+1) % 10000 == 0:
                print(f"Processed {i+1} / {all_data.shape[0]} samples")

        results['NUM_MODES'].append(NUM_MODES)
        results['DRIFT_MEAN'].append(np.mean(drift_errors))
        results['DRIFT_STD'].append(np.std(drift_errors))
        results['PCA_MEAN'].append(np.mean(pca_errors))
        results['PCA_STD'].append(np.std(pca_errors))
        results['DCT_MEAN'].append(np.mean(dct_errors))
        results['DCT_STD'].append(np.std(dct_errors))

    return results


if __name__ == "__main__":
    results = run_all_reconstructions()




import matplotlib.pyplot as plt

NUM_MODES = results['NUM_MODES']

# Colors
color_drift = "#08fb35"  # Blue
color_pca = "#0828f8"    # Orange
color_dct = "#fb0404"    # Green

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left plot: Mean MSE
axs[0].plot(NUM_MODES, results['DRIFT_MEAN'], marker='o', color=color_drift, linewidth=2, label='DRIFT')
axs[0].plot(NUM_MODES, results['PCA_MEAN'], marker='s', color=color_pca, linewidth=2, label='PCA')
axs[0].plot(NUM_MODES, results['DCT_MEAN'], marker='^', color=color_dct, linewidth=2, label='DCT')
axs[0].set_xlabel("Number of Modes / Components")
axs[0].set_ylabel("Mean Reconstruction MSE")
axs[0].set_title("Mean Reconstruction Error")
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right', frameon=True)  # Legend box enabled here

# Right plot: Std Dev MSE
axs[1].plot(NUM_MODES, results['DRIFT_STD'], marker='o', color=color_drift, linewidth=2, label='DRIFT')
axs[1].plot(NUM_MODES, results['PCA_STD'], marker='s', color=color_pca, linewidth=2, label='PCA')
axs[1].plot(NUM_MODES, results['DCT_STD'], marker='^', color=color_dct, linewidth=2, label='DCT')
axs[1].set_xlabel("Number of Modes / Components")
axs[1].set_ylabel("Std Deviation of Reconstruction MSE")
axs[1].set_title("Std Dev of Reconstruction Error")
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='upper right', frameon=True)  # Legend box enabled here

plt.tight_layout()
plt.savefig("mnist_recon_erro.png", dpi=1000)
plt.show()
