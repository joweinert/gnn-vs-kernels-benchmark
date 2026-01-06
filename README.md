# Graph Neural Networks vs. Graph Kernels: A Comparative Benchmark

This project benchmarks **Graph Neural Networks (GCN, GIN)** against classical **Graph Kernels (Weisfeiler-Lehman, Graphlet Sampling)** on the **REDDIT-BINARY** dataset.

The goal is to analyze performance in a "feature-sparse" setting where only graph topology (node degrees) is available, highlighting the trade-offs between deep learning and kernel methods.

## 📂 Project Structure

```text
P3/
├── data/                   # Dataset storage (REDDIT-BINARY)
├── results/                # Experiment logs and .pkl result files
├── src/
│   ├── data/
│   │   ├── dataset.py      # PyG Dataset Loader with caching
│   │   ├── preprocessing.py# Stratified Split generation & persistence
│   │   └── transforms.py   # OneHotDegree & Constant features
│   ├── engine/
│   │   ├── optimizer.py    # Bayesian Optimization (scikit-optimize)
│   │   └── trainer.py      # GNN Training Loop with Early Stopping
│   ├── experiments/
│   │   └── runner.py       # Cross-Validation Orchestration
│   ├── models/
│   │   ├── gnn.py          # GCN & GIN Implementations (PyG)
│   │   └── kernel.py       # WL & Graphlet Kernel Wrapper (GraKeL)
│   ├── utils/
│   │   ├── experiment_logger.py # Result pickling
│   │   └── report_generator.py  # LaTeX table & Plot generation
│   └── main.py             # CLI Entry Point
├── run_experiments.bat     # Windows Pipeline Script
├── run_experiments.sh      # Mac/Linux Pipeline Script
├── requirements.txt        # Python Dependencies
└── report.tex              # Project Report Template
```

## 🚀 Setup & Installation

We use Python 3.11 for this project (compatibility with GraKeL).

### 1. Create Virtual Environment
**Windows:**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🔬 Running Experiments

We provide automated scripts to run the full experimental pipeline (WL -> GCN -> GIN -> Ablation -> Graphlet).

### **Option A: Full Pipeline (Recommended)**
This runs all experiments sequentially and logs output to `full_experiment_log.txt`.

- **Windows:** Double-click `run_experiments.bat` or run in terminal.
- **Mac/Linux:**
  ```bash
  chmod +x run_experiments.sh
  ./run_experiments.sh
  ```

### **Option B: Manual Execution**
You can run individual experiments using the CLI:

**1. Weisfeiler-Lehman (WL) Kernel:**
```bash
python -m src.main --model Kernel --kernel_type WL --dataset REDDIT-BINARY --optimize --n_calls 20 --folds 10
```

**2. Graph Convolutional Network (GCN):**
```bash
python -m src.main --model GCN --features degree --optimize --n_calls 25 --folds 10 --epochs 50
```

**3. Graph Isomorphism Network (GIN):**
```bash
python -m src.main --model GIN --features degree --optimize --n_calls 25 --folds 10 --epochs 50
```

**4. Graphlet Sampling Kernel (GS):**
```bash
python -m src.main --model Kernel --kernel_type GS --fold 10 --k 5 --n_samples 2000
```

---

## 📊 Generating the Report
After experiments finish, results are saved in `results/`. To generate the **LaTeX table** and **Learning Curves**:

```bash
python -m src.utils.report_generator
```
This will print the LaTeX table rows to the console and save `learning_curves.png` in the results folder.

## ⚙️ Key Methodologies

- **Data Splitting:** We strictly enforce **Stratified 10-Fold Cross-Validation**. Splits are pre-calculated and persisted to `data/REDDIT-BINARY_splits.pkl` to ensure every model sees the exact same folds.
- **Optimization:** We use **Bayesian Optimization** (Gaussian Processes) to tune hyperparameters (`C`, `n_iter` for Kernels; `lr`, `hidden_dim`, `dropout`, `layers` for GNNs).
- **Metric Tracking:** We track **Accuracy**, **F1-Score (Weighted)**, **AUC**, and **Training Time**.