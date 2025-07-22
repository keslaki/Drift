# Image Classification with Feature Extraction Techniques

This repository contains Python code for training neural networks on the MNIST and CIFAR-100 datasets, comparing different feature extraction methods: Discrete Cosine Transform (DCT), Principal Component Analysis (PCA), Dynamic Response-Informed Feature Transform (DRIFT), and raw pixel inputs. The implementation uses PyTorch for neural network training and includes utilities for data preprocessing, feature extraction, and performance evaluation.

## Project Overview

The code evaluates the performance of a fully connected neural network on two standard image classification datasets:
- **MNIST**: Handwritten digit recognition (10 classes, 28x28 grayscale images).
- **CIFAR-100**: Object recognition (100 classes, 32x32 RGB images).

Four feature extraction methods are compared:
1. **DCT**: Extracts frequency components using the Discrete Cosine Transform.
2. **PCA**: Reduces dimensionality by projecting data onto principal components.
3. **DRIFT**: Uses sinusoidal mode shapes to capture spatial patterns, with padding applied for CIFAR-100.
4. **Raw**: Uses flattened pixel values without feature extraction.

The code measures preparation and training times, as well as training and validation accuracy/loss over epochs, to assess the efficiency and effectiveness of each method.

## Repository Structure

- `mnist_classification.py`: Script for training and evaluating models on the MNIST dataset.
- `cifar100_classification.py`: Script for training and evaluating models on the CIFAR-100 dataset.
- `README.md`: This file, providing an overview and instructions.

## Requirements

To run the code, install the required Python packages:

```bash
pip install torch torchvision numpy scikit-learn matplotlib scipy
```

The code is compatible with Python 3.6+ and requires a machine with or without a GPU (CUDA support is optional).

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Run the Scripts**:
   - For MNIST:
     ```bash
     python mnist_classification.py
     ```
   - For CIFAR-100:
     ```bash
     python cifar100_classification.py
     ```

3. **Output**:
   - The scripts will print:
     - Data loading and feature extraction times.
     - Training progress (loss and accuracy) for each epoch (logged every 10 epochs, plus the first and last).
     - Final training times for each feature set.
   - Training and validation metrics are stored in the `histories` variable for further analysis.

## Key Features

- **Hyperparameters**:
  - MNIST: Batch size = 512, Learning rate = 0.001, Epochs = 50, Hidden layers = [128, 256, 128], Number of modes = 49, Grid size = 28.
  - CIFAR-100: Batch size = 512, Learning rate = 0.001, Epochs = 400, Hidden layers = [64, 128, 64], Number of modes = 25, Grid size = 32, Padding for DRIFT = 5.
- **Feature Extraction**:
  - DCT: Extracts a fixed-size frequency component grid (7x7 for MNIST, 5x5 for CIFAR-100 per channel).
  - PCA: Computes principal components for dimensionality reduction (49 for MNIST, 25 per channel for CIFAR-100).
  - DRIFT: Applies sinusoidal mode shapes, with padding for CIFAR-100 to handle RGB channels.
  - Raw: Uses flattened pixel values (784 for MNIST, 3072 for CIFAR-100).
- **Neural Network**:
  - Fully connected network with ReLU activations and Adam optimizer.
  - Output layer: 10 units for MNIST, 100 units for CIFAR-100 (softmax implied via CrossEntropyLoss).
- **Reproducibility**: Fixed random seed (42) for consistent results across runs.

## Notes

- The CIFAR-100 script includes padding for DRIFT to create a larger grid (42x42) to accommodate mode shapes for RGB images.
- Feature extraction times and training times are logged to compare computational efficiency.
- The code assumes the datasets are downloaded automatically via `torchvision`. Ensure an internet connection for the first run.
- GPU support is automatically enabled if available; otherwise, it defaults to CPU.

## Results

The scripts output preparation and training times, as well as accuracy and loss metrics. To visualize results, you can extend the code to plot the `histories` data using `matplotlib`. Example metrics include:
- Training and validation accuracy/loss per epoch.
- Total time for data loading, feature extraction, and model training.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.