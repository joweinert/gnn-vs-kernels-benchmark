import argparse
import os
import torch
import warnings

from src.data.dataset import GraphDatasetLoader
from src.experiments.runner import ExperimentRunner
from src.engine.optimizer import HyperParameterOptimizer
from src.utils.experiment_logger import ResultLogger
from src.utils.utils import seed_everything
from src.models.gnn import GCN, GIN

# happens all the time in WeisfeilerLehman kernel 
# without breaking anything but clutters like crazy
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")

def main():
    parser = argparse.ArgumentParser(description="Graph Learning: GNN vs Kernels")
    parser.add_argument('--model', type=str, choices=['GCN', 'GIN', 'Kernel'], required=True, help='Model to study')
    parser.add_argument('--dataset', type=str, default='REDDIT-BINARY', help='Name of the dataset (e.g., REDDIT-BINARY, MUTAG)')
    parser.add_argument('--kernel_type', type=str, choices=['WL', 'GS'], default='WL', help='Kernel type (WL: Weisfeiler-Lehman, GS: Graphlet Sampling)')
    parser.add_argument('--features', type=str, choices=['degree', 'constant'], default='degree', help='Feature strategy')
    parser.add_argument('--optimize', action='store_true', help='Run Bayesian Optimization')
    parser.add_argument('--n_calls', type=int, default=30, help='Number of optimization calls')
    parser.add_argument('--folds', type=int, default=10, help='Number of CV folds for final evaluation')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for GNN training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # specific params (if not optimizing)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--n_iter', type=int, default=3)
    # GraphletSampling (GS) params
    parser.add_argument('--k', type=int, default=5, help='Graphlet size')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples for GS')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')
    results_dir = os.path.join(root_dir, 'results')
    
    # dataset loader
    loader = GraphDatasetLoader(root=data_dir, dataset_name=args.dataset, transform_type=args.features, seed=args.seed)
    
    # experiment runner and logger
    runner = ExperimentRunner(loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger = ResultLogger(results_dir)
    
    final_params = {}
    
    if args.optimize:
        print(f"--- Optimizing {args.model} ---")
        optimizer_class = HyperParameterOptimizer
        
        if args.model == 'Kernel':
            obj, space = runner.get_kernel_objective(kernel_type=args.kernel_type)
        else:
            obj, space = runner.get_gnn_objective(args.model, epochs=args.epochs)
        # more or less unique filenemae
        opt_filename = f"opt_{args.model}"
        if args.model == 'Kernel':
            opt_filename += f"_{args.kernel_type}"
        else:
            opt_filename += f"_{args.features}_ep{args.epochs}"
        opt_filename += f"_seed{args.seed}.pkl"
        
        output_path = os.path.join(results_dir, opt_filename)
        
        opt = optimizer_class(obj, space, n_calls=args.n_calls, output_path=output_path)
        res = opt.run()
        
        # getting best params
        # res.x is list of values in order of space dimensions
        if args.model == 'Kernel':
            if args.kernel_type == 'WL':
                final_params = {'C': float(res.x[0]), 'n_iter': int(res.x[1])}
            elif args.kernel_type == 'GS':
                final_params = {'C': float(res.x[0]), 'k': int(res.x[1]), 'n_samples': int(res.x[2])}
        else:
            final_params = {
                'lr': float(res.x[0]),
                'hidden_dim': int(res.x[1]),
                'num_layers': int(res.x[2]),
                'dropout': float(res.x[3])
            }
        print(f"Optimal Parameters: {final_params}")
    else:
        print(f"--- Running {args.model} with manual params ---")
        if args.model == 'Kernel':
            if args.kernel_type == 'WL':
                final_params = {'C': args.C, 'n_iter': args.n_iter}
            elif args.kernel_type == 'GS':
                final_params = {'C': args.C, 'k': args.k, 'n_samples': args.n_samples}
        else:
            final_params = {
                'lr': args.lr,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }

    # final evaluation with 10 folds on train val split
    print(f"--- Final Evaluation (10-Fold CV) ---")
    mean_acc, std_acc, mean_f1, std_f1, mean_auc, std_auc, mean_time = 0, 0, 0, 0, 0, 0, 0
    histories = None
    
    if args.model == 'Kernel':
        mean_acc, std_acc, mean_f1, std_f1, mean_auc, std_auc, mean_time = runner.run_kernel_cv(final_params, folds=args.folds, kernel_type=args.kernel_type)
    elif args.model == 'GCN':
        mean_acc, std_acc, mean_f1, std_f1, mean_auc, std_auc, mean_time, histories = runner.run_gnn_cv(GCN, final_params, folds=args.folds, epochs=args.epochs)
    elif args.model == 'GIN':
        mean_acc, std_acc, mean_f1, std_f1, mean_auc, std_auc, mean_time, histories = runner.run_gnn_cv(GIN, final_params, folds=args.folds, epochs=args.epochs)
        
    print(f"CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"CV F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"CV AUC:      {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Avg Training Time: {mean_time:.4f}s")
    
    # final held out test set evaluation
    test_acc, test_time, test_history = 0, 0, None
    if runner:
        print(f"--- Final Test Set Evaluation ---")
        model_type = 'Kernel' if args.model == 'Kernel' else 'GNN'
        
        # Determine class
        model_class_eval = None
        if args.model == 'GCN': model_class_eval = GCN
        elif args.model == 'GIN': model_class_eval = GIN
        
        test_acc, test_f1, test_auc, test_time, test_history = runner.evaluate_test_set(
            model_class=model_class_eval, 
            params=final_params, 
            epochs=args.epochs, 
            model_type=model_type,
            kernel_type=args.kernel_type
        )
        print(f"Test Set Accuracy: {test_acc:.4f}")
        print(f"Test Set F1-Score: {test_f1:.4f}")
        print(f"Test Set AUC:      {test_auc:.4f}")
        print(f"Test Training Time: {test_time:.4f}s")

    # saving results
    result_data = {
        'model': args.model,
        'kernel_type': args.kernel_type if args.model == 'Kernel' else None,
        'features': args.features,
        'params': final_params,
        'cv_mean_acc': mean_acc,
        'cv_std_acc': std_acc,
        'cv_mean_f1': mean_f1,
        'cv_std_f1': std_f1,
        'cv_mean_auc': mean_auc,
        'cv_std_auc': std_auc,
        'cv_mean_time': mean_time,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'test_time': test_time,
        'seed': args.seed,
        'cv_histories': histories,
        'test_history': test_history
    }
    prefix = f"FINAL_{args.model}_{args.features}"
    if args.model == 'Kernel':
        prefix = f"FINAL_{args.model}_{args.kernel_type}_{args.features}"
    logger.save(result_data, name_prefix=prefix)

if __name__ == "__main__":
    main()
