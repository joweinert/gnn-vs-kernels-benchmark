import torch
import numpy as np
import random
import os
from grakel import Graph

def seed_everything(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pyg_to_grakel(pyg_graph_list):
    """
    Converts a list of PyTorch Geometric Data objects to GraKeL Graph objects.
    We explicitly convert to grakel.Graph to avoid compatibility issues with NetworkX 3.x.
    """
    grakel_input = []
    
    for data in pyg_graph_list:
        # data.edge_index is [2, E]
        # need list of tuples (u, v)
        # convert to numpy/list
        edges = [tuple(x) for x in data.edge_index.t().tolist()]
        
        # populate node labels from data.x
        node_labels = {}
        if data.x is not None:
            # assume data.x is [NumNodes, NumFeatures] - likely OneHot
            # convert to single integer label via argmax
            labels = data.x.argmax(dim=1).tolist()
            for i, label in enumerate(labels):
                node_labels[i] = label
        
        # create GraKeL Graph
        G_grakel = Graph(edges, node_labels=node_labels)
        grakel_input.append(G_grakel)
        
    return grakel_input
