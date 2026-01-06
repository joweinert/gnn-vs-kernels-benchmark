import os
import pickle
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.data.transforms import OneHotDegree, ConstantFeature
from src.utils.utils import seed_everything

def get_max_degree(dataset):
    max_degrees = []
    for data in dataset:
        # data.num_nodes ensures we handle isolated nodes correctly if any
        d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
        max_degrees.append(d.max().item())
    
    calculated_max = max(max_degrees)
    p95 = np.percentile(max_degrees, 95)
    return calculated_max, p95


def prepare_data(root, pkl_path, dataset_name="REDDIT-BINARY", transform_type="degree", max_degree=500, seed=42):
    """
    Loads dataset, applies transforms, creates fixed splits, and pickles the result.
    Returns the path to the pickled file.
    """
    seed_everything(seed)

    print(f"Inspecting {dataset_name} topology...")
    dataset_raw = TUDataset(root=root, name=dataset_name, transform=None)
    emp_max_degree, p95 = get_max_degree(dataset_raw)
    
    print(f"Dataset Max Degree: {emp_max_degree}")
    print(f"95th Percentile: {p95}")
    
    # updates max_degree if dataset is smaller than the limit
    if emp_max_degree < max_degree:
        print(f"Optimizing max_degree: {max_degree} -> {emp_max_degree}")
        max_degree = emp_max_degree
    
    print(f"--- Preparing Data: {dataset_name} ---")
    
    transform = None
    if transform_type == "degree":
        transform = OneHotDegree(max_degree=max_degree)
    elif transform_type == "constant":
        transform = ConstantFeature(value=1.0)
        
    # loads the dataset with the transform active
    dataset = TUDataset(root=root, name=dataset_name, transform=transform)
    
    # extracts labels and indices
    y = np.array([data.y.item() for data in dataset])
    indices = np.arange(len(dataset))
    
    # strict train/test split (held-out test set)
    # uses 15% for test, 85% for training/cv
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.15, stratify=y, random_state=seed
    )
    
    print(f"Total Graphs: {len(indices)}")
    print(f"Train Set: {len(train_idx)}")
    print(f"Test Set: {len(test_idx)}")
    
    # generates 10-fold stratified splits on the training set
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv_splits = []
    
    # skf.split returns indices relative to the input array (y_train)
    # -> map back to the original dataset indices
    for fold_i, (t_idx_local, v_idx_local) in enumerate(skf.split(train_idx, y_train)):
        # t_idx_local are indices into train_idx
        train_idx_fold = train_idx[t_idx_local]
        val_idx_fold = train_idx[v_idx_local]
        
        cv_splits.append((train_idx_fold, val_idx_fold))

    # dataset itself can be re-loaded deterministically using TUDataset
    # -> we save split indices only
    save_dict = {
        'train_indices': train_idx, # full train set indices
        'test_indices': test_idx,   # held-out test set indices
        'cv_splits': cv_splits,     # list of (train, val) indices tuples for CV
        'dataset_name': dataset_name,
        'transform_type': transform_type,
        'max_degree': max_degree,
        'seed': seed
    }
    
    with open(pkl_path, "wb") as f:
        pickle.dump(save_dict, f)
        
    print(f"Data and Splits saved to {pkl_path}")
    return pkl_path

if __name__ == "__main__":
    # testing...
    root_dir = os.path.join(os.getcwd(), 'data')
    prepare_data(root_dir)
