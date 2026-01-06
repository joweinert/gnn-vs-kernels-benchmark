
import os
import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_report_assets(results_dir='results'):
    print(f"Scanning {results_dir} for results...")
    files = glob.glob(os.path.join(results_dir, "FINAL_*.pkl"))
    
    if not files:
        print("No result files found!")
        return

    results = []
    for f in files:
        with open(f, 'rb') as handle:
            data = pickle.load(handle)
            results.append(data)
            
    print("\n--- LaTeX Table Body ---")
    df = pd.DataFrame(results)
    
    # soretd by model name for consistency
    df = df.sort_values(by=['model', 'features'])
    
    for _, row in df.iterrows():
        model_name = row['model']
        feat = row['features']
        acc = row['cv_mean_acc']
        std_acc = row['cv_std_acc']
        f1 = row['cv_mean_f1']
        std_f1 = row['cv_std_f1']
        auc = row['cv_mean_auc']
        std_auc = row['cv_std_auc']
        time_sec = row['cv_mean_time']
        
        if row['model'] == 'Kernel':
            k_type = row.get('kernel_type', 'WL')
            model_display = f"Kernel ({k_type})"
        else:
            model_display = row['model']

        # Format: Model & Features & Acc & F1 & AUC & Time \\
        line = (
            f"{model_display} & {feat} & "
            f"${acc:.3f} \\pm {std_acc:.3f}$ & "
            f"${f1:.3f} \\pm {std_f1:.3f}$ & "
            f"${auc:.3f} \\pm {std_auc:.3f}$ & "
            f"{time_sec:.1f} \\\\"
        )
        print(line)
        
    print("------------------------\n")

    # Learning Curves 
    print("Generating Learning Curves plot...")
    plt.figure(figsize=(10, 6))
    
    for _, row in df.iterrows():
        if row['model'] == 'Kernel':
            continue
            
        histories = row.get('cv_histories')
        if not histories:
            continue

        # fold avg
        all_val_acc = []
        for h in histories:
            all_val_acc.append(h['val_acc'])
        
        val_acc = histories[0]['val_acc']
        plt.plot(val_acc, label=f"{row['model']} ({row['features']})")
        
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("GNN Training Progress (Fold 1)")
    plt.legend()
    plt.grid(True)
    
    output_plot = os.path.join(results_dir, "learning_curves.png")
    plt.savefig(output_plot)
    print(f"Saved plot to {output_plot}")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    res_dir = os.path.join(root, 'results')
    generate_report_assets(res_dir)
