import os
import time
import random
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

from glnn import GLNN
from gcn import GCN
from mlp import MLP
from torch_geometric.data import DataLoader, download_url, extract_zip
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from dig.xgraph.dataset import SynGraphDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BA_shapes",
        choices=["BA_shapes", "BA_Community", "Tree_Grid", "Tree_Cycle", "Cora"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
        default=0.001,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--p_dropout",
        type=float,
        default=0,
    )
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda:0")

if args.dataset == "BA_shapes" or args.dataset == "Tree_Cycle" or args.dataset == "Tree_Grid":
    # load dataset
    dataset = SynGraphDataset(root='dig_datasets/', name=args.dataset)
    dataset.data.to(device)
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    # use node degree as feature for MLP
    degrees = degree(dataset.data.edge_index[0], dtype=torch.int64)
    x = F.one_hot(degrees).float().to(device) 
    dim_node_mlp = x.shape[1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    # load trained GNN
    model = GCN(dim_node=dim_node, dim_hidden=256, num_classes=num_classes,
            n_layers=2, p_dropout=0.2)
    model.load_state_dict(torch.load(f"gnns/{args.dataset.lower()}.pt"))
    model.to(device)
elif args.dataset == "BA_Community":
    # load dataset
    dataset = SynGraphDataset(root='dig_datasets/', name='BA_Community')
    dataset.data.to(device)
    dataset.data.x = dataset.data.x.to(torch.float32)
    x = dataset.data.x
    dim_node = dataset.num_node_features
    dim_node_mlp = dim_node
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    model = GCN(dim_node=dim_node, dim_hidden=256, num_classes=num_classes, n_layers=2, p_dropout=0.2)
    model.load_state_dict(torch.load(f"gnns/ba_community.pt"))
    model.to(device)
elif args.dataset == "Cora":
    dataset = Planetoid("/tmp/Cora", name="Cora")
    dataset.data.to(device)
    dim_node = dataset.num_node_features
    x = dataset.data.x
    dim_node_mlp = dim_node
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes
    model = GCN(dim_node=dim_node, dim_hidden=256, num_classes=num_classes, n_layers=2, p_dropout=0.2)
    model.load_state_dict(torch.load(f"gnns/cora.pt"))
    model.to(device)
    
else:
    raise Exception("Unknown dataset")
# load MLP
mlp_model = MLP(num_layers=args.n_layers, input_dim=dim_node_mlp, hidden_dim=args.dim_hidden,
        output_dim=num_classes, dropout_ratio=args.p_dropout, norm_type="none")
mlp_model.to(device)
# initialize loss functions and optimizer
loss_label = torch.nn.NLLLoss()
loss_teacher = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_lambda = 0.5
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

n_samples = dataset.data.x.shape[0]

y = dataset.data.y[dataset.data.train_mask]
y_test = dataset.data.y[dataset.data.test_mask]

def evaluate():
    with torch.no_grad():
        # MLP
        mlp_model.eval()
        logits = mlp_model(x)
        z = logits[dataset.data.test_mask].log_softmax(dim=1)
        mlp_pred = z.argmax(dim=1)
        mlp_acc = float(mlp_pred.eq(y_test).sum().item()) / len(mlp_pred)
        print(f"[MLP TEST ACC]: {mlp_acc}")
        mlp_model.train()
        return mlp_acc

model.eval()
mlp_model.train()

# predict node labels with GNN
with torch.no_grad():
    logits = model(dataset.data.x, dataset.data.edge_index)
    z = logits[dataset.data.train_mask].log_softmax(dim=1)

mlp_accs = []

# train MLP
for epoch in range(args.epochs):
    optimizer.zero_grad()
    logits = mlp_model(x)
    logits = logits[dataset.data.train_mask].log_softmax(dim=1)

    # compute loss between GNN and MLP
    loss = loss_lambda * loss_label(logits, y) + (1 - loss_lambda) * loss_teacher(logits, z)

    loss.backward()
    optimizer.step()

    mlp_acc = evaluate()
    mlp_accs.append(mlp_acc)
    acc_str = str(round(mlp_acc, 3))
    print(f"[Epoch {epoch+1}] [Loss]: {loss.item()} [MLP TEST ACC]: {acc_str}")
    if mlp_acc >= max(mlp_accs):
        print(f"Found new best model at acc {acc_str}")
        torch.save(mlp_model.state_dict(), f"param/{args.dataset}_glnn.pt")

# GNN
logits = model(dataset.data.x, dataset.data.edge_index)
z = logits[dataset.data.test_mask].log_softmax(dim=1)
gnn_pred = z.argmax(dim=1)
gnn_acc = float(gnn_pred.eq(y_test).sum().item()) / len(gnn_pred)
print(f"[GNN TEST ACC]: {gnn_acc}")
print(f"[MLP BEST TEST ACC]: {max(mlp_accs)}")
print(f"Diff: {gnn_acc - max(mlp_accs)}")

data = {"accs": mlp_accs, "dataset": args.dataset, "lr": args.lr,
        "epochs": args.epochs, "dim_hidden": args.dim_hidden,
        "n_layers": args.n_layers, "p_dropout": args.p_dropout}

with open(f'results/{args.dataset}_glnn.json', 'w') as f:
    json.dump(data, f, indent=4)

# torch.save(mlp_model.state_dict(), f'param/{args.dataset}_glnn.pt')
