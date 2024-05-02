import argparse
import time
from utils import *
import pandas
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch_geometric
from torch_geometric.utils import k_hop_subgraph,to_networkx,to_undirected, subgraph
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import dgl
import networkx as nx
import netlsd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from torch_geometric.data import Data, HeteroData
# from torch_geometric.utils import from_dgl
seed_list = list(range(3407, 10000, 10))


def from_dgl(
    g: Any,
) -> Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']:
    

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    data: Union[Data, HeteroData]

    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed()




name ="reddit"

data = Dataset(name)
print(data.graph.number_of_nodes())
train_mask = data.graph.ndata['train_masks'][:,0]
train_indices = torch.nonzero(train_mask, as_tuple=False)
val_mask = data.graph.ndata['val_masks'][:,0]
val_indices = torch.nonzero(train_mask, as_tuple=False)
test_mask = data.graph.ndata['test_masks'][:,0]
test_indices = torch.nonzero(test_mask, as_tuple=False)

labels = data.graph.ndata['label']
features = data.graph.ndata.get('feature')
pyg_g = from_dgl(data.graph)
train_labels = labels[train_mask]
size = 100 ## max size of the subgraph
k=4 # the number of hops to be considered in random walk

def random_walk(edge_index, start_node, walk_length):
    walk = [start_node]
    while len(walk) < walk_length:
        neighbors = edge_index[1, edge_index[0] == walk[-1]]
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors.cpu().numpy())
        walk.append(next_node)
    return walk

def random_walk_subgraph(n, k, size, edge_index, label):
    walks = []
    for i in range(500):
        walk = random_walk(edge_index, n, k)
        walks.extend(walk)
    subset = [item[0] for item in Counter(walks).most_common(size)] if len(set(walks)) >= size else list(set(walks))

    feat = pyg_g.feature[subset]
    subg_edge_index= subgraph(subset, edge_index,relabel_nodes=True)[0]
    d = Data(x=feat, edge_index=subg_edge_index, y=label, num_nodes=len(subset))
    return d


# def get_subgraph(n,k):
#     subset, edge_index, mapping, edge_mask= k_hop_subgraph(n,k,pyg_g.edge_index, False,directed=False)
#     feat = pyg_g.feature[subset]
#     label = labels[n].item()
#     d = Data(x= feat, edge_index= edge_index,y= label, num_nodes=len(subset))
#     return d
    
def all_subgraphs(indices):
    all_g = []
    edge_index = to_undirected(pyg_g.edge_index, num_nodes=pyg_g.num_nodes)
    for train_index in indices:
        n = train_index.item()
        label = train_labels[n].item()
        d = random_walk_subgraph(n, k, size, edge_index, label)
        all_g.append(d)

    return all_g
    
def get_embedding(d):
    nx_g = to_networkx(d, to_undirected=True)
    emb = netlsd.heat(nx_g, timescales=np.logspace(-2, 2, 50))
    return emb

def get_all_emb(graphs):
    embedding_matrix = []
    for d in graphs:
        embedding_matrix.append(get_embedding(d))
    return embedding_matrix
def prepare_anomalous_data(data,clusters):
    data_list = []
    for d,c in zip(data,clusters):
        data_list.append(Data(x = d.x, edge_index=d.edge_index, cluster=c))
    return data_list

##COMPUTE SUBGRAPHS FOR THE ENTIRE DATASET:TRAIN_TEST_VAL
# train_subg = all_subgraphs(train_indices)
# val_subg = all_subgraphs(val_indices)
# test_subg = all_subgraphs(test_indices)

##compute subgraphs only for anomalous nodes

# train_anom_subgraphs_indices = torch.where(train_labels)[0]


train_anom_subgraphs_indices = torch.nonzero(train_labels == 1, as_tuple=False)
train_anom_subgraphs = all_subgraphs(train_anom_subgraphs_indices)
print(len(train_anom_subgraphs))

sizes = [d.num_nodes for d in train_anom_subgraphs]
# print(sizes)
print(np.mean(sizes))
sys.exit()


embedding_matrix = get_all_emb(train_anom_subgraphs)
embedding_matrix = np.array(embedding_matrix)


## FIND BEST K AND THEN USE WITH KMEANS
# best_score = -1
# best_k = -1
# for k in range(2, embedding_matrix.shape[0]):  
#     kmeans = KMeans(n_clusters=k)
#     labels = kmeans.fit_predict(embedding_matrix)
#     score = silhouette_score(embedding_matrix, labels)
#     if score > best_score:
#         best_score = score
#         best_k = k

## the best k is 22 WITH THE ABOVE PROCEDURE
best_k= 22 ## find the best k first and then set it here
print("Best K:", best_k)
# print("Best Silhouette Score:", best_score)

# Use the best K to cluster your data
kmeans = KMeans(n_clusters=best_k)
labels = kmeans.fit_predict(embedding_matrix)

print(len(labels))

#DATA PREPARATION TO FINETUNE DIFFUSION MODEL

anomalous_data = prepare_anomalous_data(train_anom_subgraphs,labels)
for d in anomalous_data:
    print(d)

# num_nodes = 5

# G = dgl.DGLGraph()
# G.add_nodes(5)
# G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
# G.ndata['feature'] = torch.randn(num_nodes, 2)  # Adding a node feature
# # print(G.ndata.get('feature'))
# # Get a subgraph given a node ID


# g = from_dgl(G)
# # print(g)
# # print(g.edge_index)
# # print(g.feature)
# subset = torch.tensor([0])
# subset, edge_index, mapping, edge_maskg= k_hop_subgraph(0,1,g.edge_index, False,directed=False)
# print(subset,edge_index)
# x_subg = g.feature[subset]
# print(x_subg)
# node_id = 0
# subgraph = G.subgraph([0])

# # Print the subgraph information
# print("Subgraph Edges:")
# print(subgraph.edges())
# print("\nSubgraph Node Features:")
# print(subgraph.ndata['feature'])  



