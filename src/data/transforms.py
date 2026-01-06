import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

class OneHotDegree(BaseTransform):
    """
    Adds the node degree as one-hot encoded features to the graph.
    Crucial for graphs without node features like REDDIT-BINARY to provide
    structural information to the GNN.
    """
    def __init__(self, max_degree: int = 1000):
        self.max_degree = max_degree

    def __call__(self, data: Data) -> Data:
        # degree of each node
        row, col = data.edge_index
        deg = degree(row, data.num_nodes, dtype=torch.long)
        
        # clips degree to max_degree -> fixed feature size
        deg = deg.clamp(max=self.max_degree)
        
        # one hot encode
        x = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)
        
        if data.x is not None:
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x
            
        return data

    def forward(self, data: Data) -> Data:
        return self(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_degree={self.max_degree})'


class ConstantFeature(BaseTransform):
    """
    Adds a constant value (1.0) as a feature to every node.
    Basic baseline to check if GNNs can learn purely from message passing 
    structure without distinguishing node degrees initially.
    """
    def __init__(self, value: float = 1.0):
        self.value = value

    def __call__(self, data: Data) -> Data:
        c = torch.full((data.num_nodes, 1), self.value, dtype=torch.float)
        
        if data.x is not None:
            data.x = torch.cat([data.x, c], dim=-1)
        else:
            data.x = c
            
        return data

    def forward(self, data: Data) -> Data:
        return self(data)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(value={self.value})'
