from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
import os
import pickle
from typing import List, Tuple, Generator
from src.data.preprocessing import prepare_data
from src.data.transforms import OneHotDegree, ConstantFeature

class GraphDatasetLoader:
    """
    Manages loading, preprocessing, and splitting of the REDDIT-BINARY dataset.
    Uses persisted splits from `data/processed_data.pkl` to ensure reproducibility.
    """
    def __init__(self, root: str, dataset_name: str = "REDDIT-BINARY", transform_type: str = "degree", max_degree: int = 500, seed: int = 42):
        self.root = root
        self.dataset_name = dataset_name
        self.transform_type = transform_type
        self.max_degree = max_degree
        self.seed = seed
        
        self.unique_name = f"{dataset_name}_{transform_type}"
        if transform_type == 'degree':
            self.unique_name += f"_deg{max_degree}"
        self.unique_name += f"_seed{seed}"
        self.unique_filename = f"{self.unique_name}_splits.pkl"
        self.pkl_path = os.path.join(self.root, self.unique_filename)
        
        # loads the split indices from pickle in data/
        self.data_dict = self._load_or_prepare()
        
        # loads the actual dataset (cached by TUDataset)
        self.dataset = self._load_dataset()
        
    def _load_or_prepare(self):
        """Loads indices from pickle or runs preprocessing."""
        if not os.path.exists(self.pkl_path):
            print("Processed data not found. Generating...")
            prepare_data(self.root, self.pkl_path, self.dataset_name, self.transform_type, self.max_degree, self.seed)
            
        with open(self.pkl_path, "rb") as f:
            print(f"Loading split indices from {self.pkl_path}")
            data = pickle.load(f)
                
        return data

    def _load_dataset(self) -> TUDataset:
        """Loads and pre-processes the dataset via TUDataset."""
        print(f"Loading {self.dataset_name}...")
        
        transform = None
        if self.transform_type == "degree":
            print(f"Applying OneHotDegree features (max_degree={self.max_degree})")
            transform = OneHotDegree(max_degree=self.max_degree)
        elif self.transform_type == "constant":
            print("Applying Constant features")
            transform = ConstantFeature(value=1.0)
        
        # TUDataset handles caching automatically
        dataset = TUDataset(root=self.root, name=self.dataset_name, transform=transform)
        print(f"Loaded {len(dataset)} graphs.")
        return dataset

    def get_folds(self, n_folds: int = 10, batch_size: int = 32) -> Generator[Tuple[DataLoader, DataLoader, List[int], List[int]], None, None]:
        """
        Yields (train_loader, val_loader, train_idx, val_idx) for each pre-computed fold.
        Ignores n_folds argument if it doesn't match persisted count (just warns), 
        but usually we expect 10.
        """
        cv_splits = self.data_dict['cv_splits']
        
        if len(cv_splits) < n_folds:
            raise ValueError(f"Requested {n_folds} folds but only {len(cv_splits)} are available in {self.pkl_path}")
        
        # if less folds needed
        active_splits = cv_splits[:n_folds]
        
        for fold, (train_idx, val_idx) in enumerate(active_splits):            
            # self.dataset is a list of Data objects
            # -> construct a new list for the loader
            train_subset = [self.dataset[i] for i in train_idx]
            val_subset = [self.dataset[i] for i in val_idx]
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            yield train_loader, val_loader, train_idx, val_idx
            
    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Returns the DataLoader for the held-out test set."""
        test_idx = self.data_dict['test_indices']
        test_subset = [self.dataset[i] for i in test_idx]
        return DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    def get_full_data_for_kernels(self) -> Tuple[List[object], np.ndarray]:
        """
        Returns raw data suitable for GraKeL conversion.
        """
        X = self.dataset
        y = np.array([data.y.item() for data in self.dataset])
        return X, y
