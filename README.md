# 3D Shape Classification & Point Cloud Completion

A research and educational project that implements and compares multiple deep learning architectures for **3D object classification** and **point cloud completion** using the ModelNet10 dataset.

<img width="3536" height="5000" alt="Poster" src="https://github.com/user-attachments/assets/bd0b69b4-b8a2-424d-a6ee-e366a5a9dd14" />

---

## Table of Contents

- [Overview](#overview)
- [Implemented Models](#implemented-models)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Advanced Techniques](#advanced-techniques)
- [Model Comparison](#model-comparison)

---

## Overview

This project explores four distinct deep learning paradigms for processing 3D geometric data:

1. **PointNet** – Classifies raw 3D point clouds directly using shared MLPs and global max pooling.
2. **3D CNN** – Converts point clouds into voxel grids and applies 3D convolutions for classification.
3. **DGCNN** (Dynamic Graph CNN) – Builds a dynamic k-NN graph over points and applies edge convolutions for classification.
4. **PCN** (Point Completion Network) – Reconstructs complete 3D shapes from partial/occluded point clouds.

All models are implemented in PyTorch and delivered as self-contained Jupyter Notebooks, making them easy to run, modify, and extend.

---

## Implemented Models

| Notebook | Model | Task |
|---|---|---|
| `project.ipynb` | PointNet | 3D point cloud classification |
| `3dcnn.ipynb` | 3D CNN | Voxel-based 3D classification |
| `DGCNN.ipynb` | Dynamic Graph CNN | Graph-based point cloud classification |
| `PCN.ipynb` | Point Completion Network | Partial point cloud completion |

---

## Dataset

All models are trained and evaluated on **ModelNet10**, a subset of the ModelNet40 benchmark containing **10 common object categories** (e.g., chair, table, bed, bathtub, monitor).

**Expected raw data structure:**
```
ModelNet10/
├── chair/
│   ├── train/  (*.off mesh files)
│   └── test/
├── table/
└── ... (10 classes total)
```

**Pre-processed NumPy files** (included in the repository for convenience):

| File | Description |
|---|---|
| `train_points.npy` | Training point clouds (1024 points per sample) |
| `test_points.npy` | Test point clouds |
| `train_labels.npy` | Training class labels (integers 0–9) |
| `test_labels.npy` | Test class labels |
| `train_labels_vox.npy` | Training labels for voxel-based models |
| `test_labels_vox.npy` | Test labels for voxel-based models |
| `train_acc.npy` | Training accuracy metrics |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch (`torch`, `torch.nn`, `torch.optim`) |
| Graph Neural Networks | PyTorch Geometric (`torch_geometric`) |
| 3D Data Processing | Open3D (mesh I/O, point sampling, visualization) |
| Numerical Computing | NumPy |
| Visualization | Matplotlib, Seaborn, Open3D |
| Metrics & Analysis | Scikit-learn (classification report, confusion matrix, t-SNE) |
| Environment | Jupyter Notebook, Python 3, CUDA (optional) |

---

## Project Structure

```
3dproject/
├── project.ipynb          # PointNet: point cloud classification
├── 3dcnn.ipynb            # 3D CNN: voxel-based classification
├── DGCNN.ipynb            # DGCNN: graph-based classification + focal loss
├── PCN.ipynb              # PCN: point cloud completion
├── train_points.npy       # Pre-processed training point clouds
├── test_points.npy        # Pre-processed test point clouds
├── train_labels.npy       # Training labels
├── test_labels.npy        # Test labels
├── train_labels_vox.npy   # Training labels (voxel models)
├── test_labels_vox.npy    # Test labels (voxel models)
└── train_acc.npy          # Saved training accuracy
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Azzzzen/3dproject.git
cd 3dproject
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install torch_geometric torch_cluster torch_scatter
pip install open3d
pip install numpy matplotlib scikit-learn seaborn
pip install jupyter tqdm
```

> **GPU support:** Follow the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) to install a CUDA-enabled version. The notebooks automatically detect and use a GPU if available:
> ```python
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> ```

---

## Usage

Start Jupyter Notebook and open any of the four notebooks:

```bash
jupyter notebook
```

Each notebook is self-contained and follows the same general workflow:

1. **Data Loading** – Load pre-processed `.npy` files (or regenerate from raw ModelNet10 meshes).
2. **Dataset & DataLoader** – Define PyTorch `Dataset` and `DataLoader` classes.
3. **Model Definition** – Instantiate the neural network architecture.
4. **Training Loop** – Optimize using the Adam optimizer.
5. **Evaluation** – Report test accuracy, plot confusion matrices, and visualize results.

> **Note for PCN:** Run the occlusion preprocessing cells first to generate `partial_train.npy` before training.

---

## Model Architectures

### PointNet (`project.ipynb`)

Processes raw point clouds without any spatial ordering or graph construction.

- **Input:** `N × 3` point cloud
- **Encoder:** Shared MLP via `Conv1d` layers: `3 → 64 → 128 → 1024`; BatchNorm + ReLU activations
- **Aggregation:** Global max pooling over all `N` points → `1024`-dimensional global feature
- **Classifier:** Fully connected layers: `1024 → 512 → 256 → num_classes`
- **Training:** 28 epochs, Adam optimizer (lr = 0.001), CrossEntropyLoss

### 3D CNN (`3dcnn.ipynb`)

Converts point clouds into volumetric voxel grids and applies standard 3D convolutions.

- **Input:** `64 × 64 × 64` binary occupancy grid
- **Architecture:**
  - `Conv3d(1, 32, k=3)` + `MaxPool3d(2)`
  - `Conv3d(32, 64, k=3)` + `MaxPool3d(2)`
  - `AdaptiveAvgPool3d((8, 8, 8))` — handles variable input sizes
  - FC: `64*8*8*8 → 256 → num_classes`
- **Data Pipeline:** OFF mesh → 2048-point cloud → normalize to `[0, 1]` cube → voxel grid
- **Training:** 10 epochs, batch size 16

### DGCNN — Dynamic Graph CNN (`DGCNN.ipynb`)

Constructs a dynamic k-NN graph at each layer and applies EdgeConv operations.

- **Input:** Point cloud converted to a graph using `k = 20` nearest neighbors
- **Architecture:**
  - 4 `EdgeConv` layers with dynamic edge recomputation between layers (output channels: 64, 64, 64, 128)
  - Feature concatenation: `64 + 64 + 64 + 128 = 320` dimensions
  - MLP: `320 → 1024 → 256 → num_classes`; BatchNorm + Dropout (0.5)
- **Loss:** Focal Loss with dynamic per-class weight adaptation
- **Extras:** t-SNE visualization of the learned feature space
- **Training:** 30 epochs, PyTorch Geometric `DataLoader`

### PCN — Point Completion Network (`PCN.ipynb`)

Learns to reconstruct complete point clouds from partial/occluded inputs.

- **Input:** Partial point cloud (1024 points with a box-occlusion applied along the z-axis)
- **Encoder:** PointNet-style MLP: `3 → 128 → 256 → 1024`
- **Coarse Decoder:** Global feature → 1024-point coarse output
- **Fine Decoder (Folding):** Coarse points + global feature → MLP → 4096 fine-grained points
- **Loss:** Chamfer Distance (bidirectional nearest-neighbor point set distance)
- **Training:** 30 epochs, batch size 4

---

## Advanced Techniques

| Technique | Model | Description |
|---|---|---|
| Global Max Pooling | PointNet, PCN | Permutation-invariant global feature aggregation |
| Dynamic Graph Construction | DGCNN | k-NN graph rebuilt at each layer based on learned features |
| Focal Loss | DGCNN | Down-weights easy examples to focus training on hard cases |
| Dynamic Class Weighting | DGCNN | Per-class loss weights updated based on per-class accuracy |
| Data Augmentation | DGCNN | Random z-axis rotation + Gaussian noise injection |
| Adaptive Average Pooling | 3D CNN | Handles variable spatial input sizes |
| Chamfer Distance | PCN | Symmetric geometric distance between point sets |
| t-SNE Visualization | DGCNN | 2D projection of high-dimensional learned features |

---

## Model Comparison

| Model | Input | Approach | Strength | Limitation |
|---|---|---|---|---|
| **PointNet** | Raw points | Shared MLP | Fast, permutation invariant | Ignores local structure |
| **3D CNN** | Voxel grid | 3D convolution | Structured, memory efficient | Information loss in voxelization |
| **DGCNN** | Points → Graph | Dynamic edge convolution | Captures local geometry adaptively | Higher computational cost |
| **PCN** | Partial points | Encoder-decoder + folding | Generates dense complete shapes | Requires partial/complete data pairs |
