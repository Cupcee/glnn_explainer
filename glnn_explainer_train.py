import datetime
import os
import copy
import torch
import torch.nn.functional as F
import argparse
from pgexplainer_glnn import PGExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.base_explainer import ExplainerBase
from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_self_loops, degree
from glnns.mlp import MLP
from glnns.gcn import GCN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="BA_shapes",
        choices=["BA_shapes", "BA_Community", "Tree_Grid", "Tree_Cycle"],
    )
    parser.add_argument("--train", action="store_true", default=False, help="should train")
    args = parser.parse_args()
    return args

def now():
    return datetime.datetime.utcnow().isoformat()

class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        node_idx = kwargs.get('node_idx')
        assert kwargs.get('node_idx') is not None, "please input the node_idx"
        x, edge_index, _, subset, _ = self.explainer.get_subgraph(node_idx, x, edge_index)
        self.hard_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                   device=self.device,
                                                   dtype=torch.bool)
        self.hard_edge_mask.fill_(True)

        new_node_idx = torch.where(subset == node_idx)[0]
        col, row = edge_index

        f = self.model(x).log_softmax(dim=1)
        f1 = f[col]
        f2 = f[row]
        self_embed = f[new_node_idx]
        pred, edge_mask = self.explainer.explain(x,
                                              edge_index,
                                              f1,
                                              f2,
                                              self_embed,
                                              tmp=1.0,
                                              training=False,
                                              node_idx=new_node_idx)

        # edge_masks
        edge_masks = [edge_mask for _ in range(num_classes)]
        # Calculate mask
        hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                           for _ in range(num_classes)]

        self.__clear_masks__()
        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            related_preds = self.eval_related_pred(x=x, edge_index=edge_index, edge_masks=hard_edge_masks, node_idx=new_node_idx)

        self.__clear_masks__()


        return 

# "dataset": "Tree_Cycle",
# "lr": 0.001,
# "epochs": 300,
# "dim_hidden": 300,
# "n_layers": 2,
# "p_dropout": 0.0

args = parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if args.dataset == "BA_shapes":
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
    hidden_dim = 300
    num_classes = dataset.num_classes
    # load trained GNN
    state_dict = torch.load(f"glnns/gnns/{args.dataset.lower()}.pt")
    model = GCN(dim_node=dim_node, dim_hidden=300, num_classes=num_classes,
                n_hidden_layers=2, p_dropout=0)
    model.load_state_dict(state_dict)
    model.to(device)

elif args.dataset == "Tree_Cycle":
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
    hidden_dim = 300
    num_classes = dataset.num_classes
    # load trained GNN
    state_dict = torch.load(f"glnns/gnns/{args.dataset.lower()}.pt")
    model = GCN(dim_node=dim_node, dim_hidden=300, num_classes=num_classes,
                n_hidden_layers=2, p_dropout=0)
    model.load_state_dict(state_dict)
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
    hidden_dim = 300
    num_classes = dataset.num_classes
    state_dict = torch.load(f"glnns/gnns/ba_community.pt")
    model = GCN(dim_node=dim_node, dim_hidden=hidden_dim, num_classes=num_classes,
                n_hidden_layers=8, p_dropout=0.1)
    model.load_state_dict(state_dict)
    model.to(device)
else:
    raise Exception("Unknown dataset")

# load MLP
glnn = MLP(num_layers=2, input_dim=dim_node_mlp, hidden_dim=hidden_dim,
        output_dim=num_classes, dropout_ratio=0.0, norm_type="none")

glnn.load_state_dict(torch.load(f"glnns/param/{args.dataset}_glnn.pt", map_location=device))
explainer = PGExplainer(
    glnn,
    in_channels=num_classes * 3, # 20?
    device=device,
    explain_graph=False,
    epochs=20,
    lr=3e-3,
    coff_size=0.01,
    coff_ent=5e-4,
    t0=5.0,
    t1=1.0,
    num_hops=3,
)

# train
if args.train:
    print(f"Training for dataset {args.dataset}...")
    dataset = copy.copy(dataset[0])
    dataset.x = x
    explainer.train_explanation_network(dataset)
    torch.save(explainer.state_dict(), "models/pg_explainer.pt")

print("Loading state dict")
explainer.load_state_dict(torch.load("models/pg_explainer.pt"))

# print("Evaluating...")
# 
# dataset = copy.copy(dataset[0])
# dataset.x = x
# accuracy = explainer.test_explanation_network(dataset)
# print(f"Pred acc: {accuracy}")
