@echo off
echo ========================================================
echo STARTING FULL EXPERIMENTAL PIPELINE
echo Date: %date% %time%
echo Logging all output to full_experiment_log.txt
echo ========================================================

echo EXPERIMENT LOG > full_experiment_log.txt
echo Started at %date% %time% >> full_experiment_log.txt
echo -------------------------------------------------------- >> full_experiment_log.txt

echo [1/5] Running WL Kernel Baseline...
echo [1/5] Running WL Kernel Baseline... >> full_experiment_log.txt
python -m src.main --model Kernel --kernel_type WL --dataset REDDIT-BINARY --optimize --n_calls 20 --folds 10 >> full_experiment_log.txt 2>&1
echo DONE. >> full_experiment_log.txt

echo [2/5] Running GCN...
echo [2/5] Running GCN... >> full_experiment_log.txt
python -m src.main --model GCN --features degree --optimize --n_calls 25 --folds 10 --epochs 50 >> full_experiment_log.txt 2>&1
echo DONE. >> full_experiment_log.txt

echo [3/5] Running GIN...
echo [3/5] Running GIN... >> full_experiment_log.txt
python -m src.main --model GIN --features degree --optimize --n_calls 25 --folds 10 --epochs 50 >> full_experiment_log.txt 2>&1
echo DONE. >> full_experiment_log.txt

echo [4/5] Running GIN Ablation (Constant Features)...
echo [4/5] Running GIN Ablation (Constant Features)... >> full_experiment_log.txt
python -m src.main --model GIN --features constant --optimize --n_calls 15 --folds 10 --epochs 50 >> full_experiment_log.txt 2>&1
echo DONE. >> full_experiment_log.txt

echo [5/5] Running Graphlet Kernel...
echo [5/5] Running Graphlet Kernel... >> full_experiment_log.txt
REM Note: No optimization here, just a manual run with k=5 to verify computational limits
python -m src.main --model Kernel --kernel_type Graphlet --k 5 --n_samples 2000 --C 1.0 --folds 10 >> full_experiment_log.txt 2>&1
echo DONE. >> full_experiment_log.txt

echo ========================================================
echo PIPELINE FINISHED
echo Date: %date% %time%
echo Check full_experiment_log.txt for results.
echo ========================================================
pause