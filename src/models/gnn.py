import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.pool import global_add_pool, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=3, pooling='mean', dropout=0.5):
        super(GCN, self).__init__()
        self.pooling = pooling
        self.dropout_rate = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
            
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
            
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=3, pooling='add', dropout=0.5):
        super(GIN, self).__init__()
        self.pooling = pooling
        self.dropout_rate = dropout
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = num_node_features
            else:
                input_dim = hidden_dim
            
            # MLP: Linear -> BN -> ReLU -> Linear
            # (GINConv expects a neural network)
            # mapping input_dim -> hidden_dim -> hidden_dim
            mlp = Sequential(
                Linear(input_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_dim))

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
            
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)