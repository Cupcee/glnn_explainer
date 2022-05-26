import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, dim_node: int, dim_hidden: int, num_classes: int, n_layers: int, p_dropout: float):
        super().__init__()
        # self.conv1 = GCNConv(dim_node, dim_hidden)
        self.layers = torch.nn.ModuleList([
                GCNConv(dim_node if i == 0 else dim_hidden, dim_hidden) for i in range(n_layers - 1)
        ])
        self.head = GCNConv(dim_hidden, num_classes)
        # self.ffn = torch.nn.Sequential(*(
        #             [torch.nn.Linear(dim_hidden, num_classes)]
        # ))
        self.p_dropout = p_dropout
 
    def _argsparse(self, *args, **kwargs):
        r""" Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]

            else:
                raise ValueError(f"forward's args should take 1, 2 or 3 arguments but got {len(args)}")
        else:
            data: Batch = kwargs.get('data')
            if not data:
                x = kwargs.get('x')
                edge_index = kwargs.get('edge_index')
                assert x is not None, "forward's args is empty and required node features x is not in kwargs"
                assert edge_index is not None, "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get('batch')
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, 'batch'):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, batch


    # def get_emb(self, *args, **kwargs):
    #     x, edge_index, _ = self._argsparse(*args, **kwargs)
    #     for i in range(len(self.hidden)):
    #         x = self.hidden[i](x, edge_index)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=self.p_dropout, training=self.training)
    #     x = F.relu(x)
    #     return x

    def get_emb(self, *args, **kwargs):
        x, edge_index, _ = self._argsparse(*args, **kwargs)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.head(x, edge_index) 
        return x

    def forward(self, x, edge_index, **kwargs):
        out = self.get_emb(x=x, edge_index=edge_index, **kwargs)
        return out


