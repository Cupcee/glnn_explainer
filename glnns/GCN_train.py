import os
import time
import random
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import DataLoader, download_url, extract_zip
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import GCN_2l, GCN_3l
from torch_geometric.nn import GCNConv
from gcn import GCN

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BA_shapes",
        choices=["BA_shapes", "BA_Community", "Tree_Cycle", "Tree_Grid"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--dim_hidden",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--n_hidden_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--p_dropout",
        type=float,
        default=0,
    )
    return parser.parse_args()


args = parse_args()
device = torch.device("cuda:0")
# load dataset
dataset = SynGraphDataset(root='dig_datasets/', name=args.dataset)
print(f"Using dataset {args.dataset}")
dataset.data.x = dataset.data.x.to(torch.float32)
if args.dataset != "BA_Community":
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = 1
else:
    dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes
model = GCN(dim_node=dim_node, dim_hidden=args.dim_hidden, num_classes=num_classes,
        n_hidden_layers=args.n_hidden_layers, p_dropout=args.p_dropout)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.CrossEntropyLoss()

data = dataset[0]
data.to(device)
y = data.y[data.train_mask]
y_test = data.y[data.test_mask]
n_test = int(data.test_mask.sum())
accs = []
for i in range(args.epochs):
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)

    loss = loss_fn(logits[data.train_mask], y) 

    loss.backward()
    optimizer.step()
    
    if i % 20 == 0:
        with torch.no_grad():
            model.eval()
            pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = float(pred[data.test_mask].eq(y_test).sum().item()) / n_test
            accs.append(acc)
            print(f"[GCN TEST ACC]: {round(acc, 3)}")
            model.train()

data = {"accs": accs, "dataset": args.dataset, "lr": args.lr,
        "epochs": args.epochs, "dim_hidden": args.dim_hidden,
        "n_hidden_layers": args.n_hidden_layers, "p_dropout": args.p_dropout}
with open(f"results/{args.dataset}_accs.json", "w") as f:
    json.dump(data, f, indent=4)
torch.save(model, f"gnns/{args.dataset}.pt")
