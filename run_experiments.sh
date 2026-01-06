#!/bin/bash

LOGFILE="full_experiment_log.txt"

echo "========================================================" | tee -a "$LOGFILE"
echo "STARTING FULL EXPERIMENTAL PIPELINE" | tee -a "$LOGFILE"
echo "Date: $(date)" | tee -a "$LOGFILE"
echo "Logging all output to $LOGFILE" | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"

# 1. WL Kernel Baseline
echo "[1/5] Running WL Kernel Baseline..." | tee -a "$LOGFILE"
python3 -m src.main --model Kernel --kernel_type WL --dataset REDDIT-BINARY --optimize --n_calls 20 --folds 10 >> "$LOGFILE" 2>&1
echo "DONE." | tee -a "$LOGFILE"

# 2. GCN
echo "[2/5] Running GCN..." | tee -a "$LOGFILE"
python3 -m src.main --model GCN --features degree --optimize --n_calls 25 --folds 10 --epochs 50 >> "$LOGFILE" 2>&1
echo "DONE." | tee -a "$LOGFILE"

# 3. GIN
echo "[3/5] Running GIN..." | tee -a "$LOGFILE"
python3 -m src.main --model GIN --features degree --optimize --n_calls 25 --folds 10 --epochs 50 >> "$LOGFILE" 2>&1
echo "DONE." | tee -a "$LOGFILE"

# 4. GIN Ablation (Constant Features)
echo "[4/5] Running GIN Ablation (Constant Features)..." | tee -a "$LOGFILE"
python3 -m src.main --model GIN --features constant --optimize --n_calls 15 --folds 10 --epochs 50 >> "$LOGFILE" 2>&1
echo "DONE." | tee -a "$LOGFILE"

# 5. Graphlet Kernel
echo "[5/5] Running Graphlet Kernel..." | tee -a "$LOGFILE"
# Note: No optimization here, just a manual run with k=5
python3 -m src.main --model Kernel --kernel_type Graphlet --k 5 --n_samples 2000 --C 1.0 --folds 10 >> "$LOGFILE" 2>&1
echo "DONE." | tee -a "$LOGFILE"

echo "========================================================" | tee -a "$LOGFILE"
echo "PIPELINE FINISHED" | tee -a "$LOGFILE"
echo "Date: $(date)" | tee -a "$LOGFILE"
echo "Check $LOGFILE for results." | tee -a "$LOGFILE"
echo "========================================================" | tee -a "$LOGFILE"