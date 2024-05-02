import pickle
import torch

anomaly_data = torch.load('reddit_anomalous_data.pt')['anomalous_data']

# get the distribution of the number of nodes in the dataset
node_num_dist = {}
for data in anomaly_data:
    num_nodes = data.num_nodes
    if num_nodes in node_num_dist:
        node_num_dist[num_nodes] += 1
    else:
        node_num_dist[num_nodes] = 1
        
with open('./samples/pkl/anomaly/test/gdss_anomaly-sample.pkl', 'rb') as f:
    sample_anomaly = pickle.load(f)

# get the distribution of the number of nodes in the sample
sample_node_num_dist = {}
for G in sample_anomaly:
    num_nodes = G.number_of_nodes()
    if num_nodes in sample_node_num_dist:
        sample_node_num_dist[num_nodes] += 1
    else:
        sample_node_num_dist[num_nodes] = 1


# visualize the two distributions in a bar plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.bar(node_num_dist.keys(), node_num_dist.values(), alpha=0.5, label='Anomaly Data')
plt.bar(sample_node_num_dist.keys(), sample_node_num_dist.values(), alpha=0.5, label='Sample Anomaly')
plt.xlabel('Number of Nodes')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Nodes in Anomaly Data and Sample Anomaly')
plt.legend()
plt.grid()
plt.savefig('node_num_dist.png')
plt.show()


# ----------------------------

# get the distribution of node features in the dataset
feature_list = []
for data in anomaly_data:
    features = data.x.numpy().ravel().tolist()
    feature_list.extend(features)
    
# get the distribution of node features in the sample

sample_feature_list = []

for G in sample_anomaly:
    for node in G.nodes(data=True):
        if 'feature' in node[1]:  # Check if 'feature' key exists
            feature = node[1]['feature']  # Access the 'feature' attribute
            sample_feature_list.extend(feature)
        
    

# visualize both distribution of node features in the dataset and sample
plt.figure(figsize=(10, 5))
plt.hist(feature_list, bins=50, alpha=0.5, label='Anomaly Data')
plt.hist(sample_feature_list, bins=50, alpha=0.5, label='Sample Anomaly')
plt.xlabel('Node Features')
plt.ylabel('Frequency')
plt.title('Distribution of Node Features in Anomaly Data and Sample Anomaly')
plt.legend()
plt.grid()
plt.savefig('node_feature_dist.png')
plt.show()



            

