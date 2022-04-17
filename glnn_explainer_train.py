import os
import torch
import torch.nn.functional as F
from pgexplainer_glnn import PGExplainer
from dig.xgraph.evaluation import XCollector
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.method.base_explainer import ExplainerBase
from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_self_loops, degree
from glnns.mlp import MLP

class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x: Tensor,
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

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)

        else:
            node_idx = kwargs.get('node_idx')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            x, edge_index, _, subset, _ = self.explainer.get_subgraph(node_idx, x, edge_index)
            self.hard_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)

            new_node_idx = torch.where(subset == node_idx)[0]
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
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
            if self.explain_graph:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)
            else:
                related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks,
                                                       node_idx=new_node_idx)

        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds 


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = SynGraphDataset('./glnns/dig_datasets', 'BA_shapes')
dataset.data.x = dataset.data.x.float()
dataset.data.y = dataset.data.y.squeeze().long()
dataset.data.to(device)
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes
degrees = degree(dataset.data.edge_index[0], dtype=torch.int64)
x = F.one_hot(degrees).float().to(device)
dataset.data.x = x
n_samples, n_features = x.shape 

glnn = MLP(num_layers=2, input_dim=n_features, hidden_dim=300,
        output_dim=num_classes, dropout_ratio=0, norm_type="none")
glnn.load_state_dict(torch.load("glnns/param/BA_shapes_glnn.pt", map_location=device))
explainer = PGExplainer(
    glnn,
    in_channels=n_features, # 20?
    device=device,
    explain_graph=False,
    epochs=20,
    lr=3e-3,
    coff_size=0.01,
    coff_ent=5e-4,
    t0=5.0,
    t1=1.0
)

# train
explainer.train_explanation_network(dataset)
torch.save(pgexplainer.state_dict(), '')

index = 0
x_collector = XCollector()
pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer,
                                      model=eval_model,
                                      molecule=True)

pgexplainer_edges.device = pgexplainer.device
node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
predictions = model(data).softmax(dim=-1).argmax(dim=-1)
for node_idx in node_indices:
    index += 1
    data.to(device)
    with torch.no_grad():
        edge_masks, hard_edge_masks, related_preds = \
            pgexplainer_edges(data.x, data.edge_index,
                              node_idx=node_idx,
                              num_classes=dataset.num_classes,
                              sparsity=0.5,
                              pred_label=predictions[node_idx].item())
        edge_masks = [mask.detach() for mask in edge_masks]
    x_collector.collect_data(edge_masks, related_preds, label=predictions[node_idx].item()) 

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
