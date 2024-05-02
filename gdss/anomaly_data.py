import torch 
import pickle

import networkx as nx

def pyg_to_networkx(data):
    G = nx.Graph()

    # First, extract all unique node IDs from edge_index
    unique_nodes = torch.unique(data.edge_index).tolist()

    # Assume features in `x` are aligned with these sorted unique node IDs
    features = {node: data.x[i] for i, node in enumerate(sorted(unique_nodes))}

    # Add nodes with features
    for node in unique_nodes:
        G.add_node(node, features=features[node])

    # Add edges
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        G.add_edge(src, dst)

    return G


dataset = torch.load('reddit_anomalous_data.pt')['anomalous_data']
print(len(dataset))

# get the max number of nodes in the dataset
max_node_num = 0
for data in dataset:
    num_nodes = data.num_nodes
    if num_nodes > max_node_num:
        max_node_num = num_nodes
        
print(max_node_num)
print(dataset[0].x.shape)



# aomaly_list = []
# for data in dataset:
#     G = pyg_to_networkx(data)
#     aomaly_list.append(G)
#     print(G.nodes(data=True))
#     print(G.edges())

# with open('data/anomaly.pkl', 'wb') as f:
#     pickle.dump(aomaly_list, f)



# ----------------------------
import pickle


with open('./samples/pkl/anomaly/test/gdss_anomaly-sample.pkl', 'rb') as f:
    sample_anomaly = pickle.load(f)

i = 3
print(sample_anomaly[i].nodes(data=True))
print(sample_anomaly[i].edges())

