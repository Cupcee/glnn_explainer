import datetime
import os
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
        _, edge_mask = self.explainer.explain(x,
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

        return edge_masks, hard_edge_masks, related_preds 


args = parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    hidden_dim = 300
    num_classes = dataset.num_classes
    # load trained GNN
    model = torch.load(f"gnns/{args.dataset}.pt")
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
    model = torch.load('gnns/BA_Community.pt')
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
    print("Training...")
    explainer.train_explanation_network(dataset)
    torch.save(explainer.state_dict(), "models/pg_explainer.pt")

print("Loading state dict")
explainer.load_state_dict(torch.load("models/pg_explainer.pt"))

index = 0
x_collector = XCollector()
pgexplainer_edges = PGExplainer_edges(pgexplainer=explainer,
                                      model=glnn,
                                      molecule=True)

print("Evaluating...")
pgexplainer_edges.device = explainer.device
data = dataset[0]
node_indices = torch.where(data.test_mask * data.y != 0)[0].tolist()
predictions = glnn(x).log_softmax(dim=-1).argmax(dim=-1)
for node_idx in node_indices:
    index += 1
    with torch.no_grad():
        edge_masks, hard_edge_masks, related_preds = \
            pgexplainer_edges(x, 
                              edge_index=data.edge_index,
                              node_idx=node_idx,
                              num_classes=dataset.num_classes,
                              sparsity=0.5,
                              pred_label=predictions[node_idx].item())
        edge_masks = [mask.detach() for mask in edge_masks]
    x_collector.collect_data(edge_masks, related_preds, label=predictions[node_idx].item()) 

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
