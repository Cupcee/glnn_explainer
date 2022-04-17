import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, dim_node, dim_hidden, num_classes, n_hidden_layers, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.hidden = torch.nn.ModuleList([
            GCNConv(dim_hidden, dim_hidden) for _ in range(n_hidden_layers)
        ])
        self.ffn = torch.nn.Sequential(*(
                    [torch.nn.Linear(dim_hidden, num_classes)]
        ))
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(x)
        out = self.ffn(x)
        return out


