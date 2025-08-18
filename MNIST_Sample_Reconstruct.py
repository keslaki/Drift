import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

GRID_SIZE = 28
POR = 1.0
USE_SAMPLE_NORM = True

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
    modes = [np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly) for m,n in pairs]
    return np.array([m.flatten() for m in modes])

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
    weights = sample_drift * sample_norm * modes_norms
    projection_coeffs = weights / (modes_norms**2)
    return projection_coeffs @ modes

def reconstruct_from_pca(sample_pca, pca, scaler):
    x_scaled_recon = pca.inverse_transform(sample_pca.reshape(1, -1))
    return scaler.inverse_transform(x_scaled_recon).flatten()

def reconstruct_from_dct(sample_dct, final_side, grid_size):
    dct_2d = np.zeros((grid_size, grid_size))
    dct_2d[:final_side, :final_side] = sample_dct.reshape(final_side, final_side)
    return idct(idct(dct_2d.T, norm='ortho').T, norm='ortho').flatten()

if __name__ == "__main__":
    print("Loading MNIST data...")
    all_data = load_mnist_all(POR)
    print(f"Total samples: {all_data.shape[0]}")

    sample_idx = 0  # Pick first sample
    sample = all_data[sample_idx]
    norm_orig = np.linalg.norm(sample)
    sample_norm = norm_orig if USE_SAMPLE_NORM else 1.0

    fig, axs = plt.subplots(len(NUM_MODES_LIST), 4, figsize=(8, 8))
    methods = ["Original", "DRIFT", "PCA", "DCT"]

    for row, (NUM_MODES, DCT_FINAL_SIDE) in enumerate(zip(NUM_MODES_LIST, DCT_FINAL_SIDES)):
        print(f"Processing NUM_MODES={NUM_MODES}, DCT_FINAL_SIDE={DCT_FINAL_SIDE}")

        modes = generate_mode_shapes(GRID_SIZE, NUM_MODES)
        x_drift = compute_drift_features(all_data, modes)
        x_pca, pca, scaler = compute_pca_features(all_data, NUM_MODES)
        x_dct = compute_dct_features(all_data, DCT_FINAL_SIDE, GRID_SIZE)

        drift_recon = reconstruct_from_drift(x_drift[sample_idx], modes, sample_norm=sample_norm)
        pca_recon = reconstruct_from_pca(x_pca[sample_idx], pca, scaler)
        dct_recon = reconstruct_from_dct(x_dct[sample_idx], DCT_FINAL_SIDE, GRID_SIZE)

        recons = [sample, drift_recon, pca_recon, dct_recon]
        for col, img in enumerate(recons):
            axs[row, col].imshow(img.reshape(GRID_SIZE, GRID_SIZE), cmap="gray")
            axs[row, col].axis("off")
            if row == 0:
                axs[row, col].set_title(methods[col])

    plt.tight_layout()
    plt.savefig("mnist_single_recon.png", dpi=700)
    plt.show()
