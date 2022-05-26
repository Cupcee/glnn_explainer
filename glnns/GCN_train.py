import os
import time
import random
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import DataLoader, download_url, extract_zip
from torch_geometric.datasets import Planetoid
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
        choices=["BA_shapes", "BA_Community", "Tree_Cycle", "Tree_Grid", "Cora"],
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
        "--n_layers",
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
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
    )
    return parser.parse_args()


args = parse_args()
device = torch.device("cuda:0")

# load dataset
if args.dataset == "Cora":
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
else:
    dataset = SynGraphDataset(root='dig_datasets/', name=args.dataset)

print(f"Using dataset {args.dataset}")
dataset.data.x = dataset.data.x.to(torch.float32)
if args.dataset == "BA_shapes" or args.dataset == "Tree_Cycle" or args.dataset == "Tree_Grid":
    dataset.data.x = dataset.data.x[:, :1]
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    dim_node = 1
    model = GCN(dim_node=dim_node,
            dim_hidden=args.dim_hidden,
            num_classes=num_classes,
            n_layers=args.n_layers,
            p_dropout=args.p_dropout)
elif args.dataset == "Cora":
    dim_node = dataset.num_node_features
    dim_node_mlp = dim_node
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    model = GCN(dim_node=dim_node,
            dim_hidden=args.dim_hidden,
            num_classes=num_classes,
            n_layers=args.n_layers,
            p_dropout=args.p_dropout)
elif args.dataset == "BA_Community":
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    dim_node = dataset.num_node_features
    model = GCN(dim_node=dim_node,
            dim_hidden=args.dim_hidden,
            num_classes=num_classes,
            n_layers=args.n_layers,
            p_dropout=args.p_dropout)
else:
    raise Exception(f"Unknown dataset {args.dataset}")

model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            acc_str = str(round(acc, 3))
            print(f"[GCN TEST ACC]: {acc_str}")
            if acc >= max(accs):
                print(f"Found new best model at acc {acc_str}")
                torch.save(model.state_dict(), f"gnns/{args.dataset.lower()}.pt")
            model.train()
                        

data = {"accs": accs, "dataset": args.dataset, "lr": args.lr,
        "epochs": args.epochs, "dim_hidden": args.dim_hidden,
        "n_layers": args.n_layers, "p_dropout": args.p_dropout}
with open(f"results/{args.dataset}_accs.json", "w") as f:
    json.dump(data, f, indent=4)
#torch.save(model.state_dict(), f"gnns/{args.dataset}.pt")
