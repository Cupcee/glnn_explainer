import torch.nn as nn

class GLNN(nn.Module):
    def __init__(self, device, num_layers, in_features, out_features, dim_hidden):
        super().__init__()
        self.num_hidden_layers = num_layers - 1
        self.node_emb = nn.Sequential(
            nn.Linear(in_features, dim_hidden), # node embedding layer
            nn.ReLU(),
        ).to(device)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU()
            ) for _ in range(self.num_hidden_layers)
        ]).to(device)
        self.out = nn.Linear(dim_hidden, out_features).to(device)

    def forward(self, x):
        x = self.node_emb(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
        return None, self.out(x)
