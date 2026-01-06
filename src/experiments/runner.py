import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.data.dataset import GraphDatasetLoader
from src.models.gnn import GCN, GIN
from src.models.kernel import KernelSVC
from src.engine.trainer import GNNTrainer
from src.utils.utils import pyg_to_grakel
from skopt.space import Real, Integer
import time

class ExperimentRunner:
    def __init__(self, data_loader: GraphDatasetLoader, device=None):
        self.data_loader = data_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._grakel_cache = None

    def _get_grakel_data(self):
        if self._grakel_cache is None:
            X, y = self.data_loader.get_full_data_for_kernels()
            print("Converting all graphs to Grakel format (Cached)...")
            X_grakel = pyg_to_grakel(X)
            X_grakel = np.array(X_grakel, dtype=object)
            self._grakel_cache = (X_grakel, y)
        return self._grakel_cache

    def run_gnn_cv(self, model_class, params, folds=10, epochs=50):
        val_accuracies = []
        val_f1s = []
        val_aucs = []
        train_times = []
        histories = []
        
        fold_gen = self.data_loader.get_folds(n_folds=folds, batch_size=32)
        
        for i, (train_loader, val_loader, _, _) in enumerate(fold_gen):
            sample_data = self.data_loader.dataset[0]
            num_features = sample_data.x.shape[1]
            num_classes = self.data_loader.dataset.num_classes
            
            model = model_class(
                num_node_features=num_features,
                hidden_dim=params['hidden_dim'],
                num_classes=num_classes,
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            trainer = GNNTrainer(model, optimizer, criterion, device=self.device)
            val_acc_best, history, t_time = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False, restore_best=False)
            
            final_acc = history['val_acc'][-1]
            final_f1 = history['val_f1'][-1]
            final_auc = history['val_auc'][-1]

            val_accuracies.append(final_acc)
            val_f1s.append(final_f1)
            val_aucs.append(final_auc)
            train_times.append(t_time)
            
            print(f" Fold {i+1}/{folds} | Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

            history['fold'] = i
            histories.append(history)
            
        return np.mean(val_accuracies), np.std(val_accuracies), np.mean(val_f1s), np.std(val_f1s), np.mean(val_aucs), np.std(val_aucs), np.mean(train_times), histories

    def run_kernel_cv(self, params, folds=10, kernel_type='WL'):
        X_grakel, y = self._get_grakel_data()
 
        
        val_accuracies = []
        val_f1s = []
        val_aucs = []
        train_times = []
        
        fold_gen = self.data_loader.get_folds(n_folds=folds, batch_size=32)
        
        for i, (_, _, train_idx, val_idx) in enumerate(fold_gen):
            X_train = list(X_grakel[train_idx])
            y_train = y[train_idx]
            X_test = list(X_grakel[val_idx])
            y_test = y[val_idx]
            
            # GS specific params if needed
            k = params.get('k', 5)
            n_samples = params.get('n_samples', 100)
            n_iter = params.get('n_iter', 5)
            
            model = KernelSVC(
                kernel_type=kernel_type, 
                C=params['C'], 
                n_iter=n_iter,
                k=k,
                n_samples=n_samples
            )
            
            start_t = time.perf_counter()
            model.fit(X_train, y_train)
            acc, f1, auc = model.get_metrics(X_test, y_test)
            
            end_t = time.perf_counter()
            val_accuracies.append(acc)
            val_f1s.append(f1)
            val_aucs.append(auc)
            
            print(f" Fold {i+1}/{folds} | Acc: {acc:.4f} | F1: {f1:.4f}")
            
            train_times.append(end_t - start_t)
            
        return np.mean(val_accuracies), np.std(val_accuracies), np.mean(val_f1s), np.std(val_f1s), np.mean(val_aucs), np.std(val_aucs), np.mean(train_times)

    def get_gnn_objective(self, model_name, epochs=50):
        space = [
            Real(1e-4, 1e-2, prior='log-uniform', name='lr'),
            Integer(16, 64, name='hidden_dim'),
            Integer(2, 5, name='num_layers'),
            Real(0.0, 0.5, name='dropout')
        ]
        
        def objective(lr, hidden_dim, num_layers, dropout):
            model_class = GIN if model_name == 'GIN' else GCN
            params = {
                'lr': lr,
                'hidden_dim': int(hidden_dim),
                'num_layers': int(num_layers),
                'dropout': dropout
            }
            print(f"Testing params: {params}")
            results = self.run_gnn_cv(model_class, params, folds=5, epochs=epochs) 
            mean_acc = results[0]
            
            return -mean_acc 
            
        return objective, space

    def get_kernel_objective(self, kernel_type='WL'):
        if kernel_type == 'WL':
            space = [
                Real(1e-2, 1e2, prior='log-uniform', name='C'),
                Integer(2, 7, name='n_iter')
            ]
            
            def objective(C, n_iter):
                params = {'C': C, 'n_iter': int(n_iter)}
                print(f"Testing params: {params}")
                results = self.run_kernel_cv(params, folds=5, kernel_type='WL')
                mean_acc = results[0]
                return -mean_acc
                
        elif kernel_type == 'GS':
            space = [
                Real(1e-2, 1e2, prior='log-uniform', name='C'),
                Integer(3, 5, name='k'),
                Integer(2000, 5000, name='n_samples')
            ]
            
            def objective(C, k, n_samples):
                params = {'C': C, 'k': int(k), 'n_samples': int(n_samples)}
                print(f"Testing params: {params}")
                results = self.run_kernel_cv(params, folds=5, kernel_type='GS')
                mean_acc = results[0]
                return -mean_acc
        else:
             raise ValueError(f"Unknown kernel type: {kernel_type}")
             
        return objective, space

    def evaluate_test_set(self, model_class, params, epochs=50, model_type='GNN', kernel_type='WL'): # Add kernel_type
        print("\n--- Final Test Set Evaluation ---")
        
        full_train_idx = self.data_loader.data_dict['train_indices']
        test_idx = self.data_loader.data_dict['test_indices']
        
        if model_type == 'Kernel':
            X_grakel, y = self._get_grakel_data()
            print("Using cached Grakel data for Test Eval...")
            
            X_train = list(X_grakel[full_train_idx])
            y_train = y[full_train_idx]
            X_test = list(X_grakel[test_idx])
            y_test = y[test_idx]
            
            k = params.get('k', 5)
            n_samples = params.get('n_samples', 100)
            n_iter = params.get('n_iter', 5)
            
            model = KernelSVC(
                kernel_type=kernel_type,
                C=params['C'], 
                n_iter=n_iter,
                k=k,
                n_samples=n_samples
            )
            
            start_t = time.perf_counter()
            model.fit(X_train, y_train)
            acc, f1, auc = model.get_metrics(X_test, y_test)
            test_time = time.perf_counter() - start_t
            
            return acc, f1, auc, test_time, None
            
        else: # GNN
            dataset_list = self.data_loader.dataset
            train_subset = [dataset_list[i] for i in full_train_idx]
            
            # loader for the FULL training set
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            test_loader = self.data_loader.get_test_loader(batch_size=32)
            
            sample_data = dataset_list[0]
            num_features = sample_data.x.shape[1]
            num_classes = int(np.max([d.y.item() for d in dataset_list]) + 1)

            model = model_class(
                num_node_features=num_features,
                hidden_dim=params['hidden_dim'],
                num_classes=num_classes,
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            trainer = GNNTrainer(model, optimizer, criterion, device=self.device, patience=epochs)
            
            print("Retraining on full training set...")
            val_acc, history, t_time = trainer.fit(train_loader, test_loader, epochs=epochs, verbose=False, restore_best=False)            
            final_acc = history['val_acc'][-1]
            final_f1 = history['val_f1'][-1]
            final_auc = history['val_auc'][-1]
            
            return final_acc, final_f1, final_auc, t_time, history