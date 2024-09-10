import networkx as nx
import numpy as np
import pandas as pd
import random

#This function creates a fully connected graph using the input features.
#Parameters:
#features: A list of features, where each element is an array (or list) representing the feature of a node.
def create_full_graph(features):

    G = nx.Graph()
    for i, feature in enumerate(features):
        G.add_node(i, feature=np.array(feature))
    for i in G.nodes:
        for j in G.nodes:
            if i != j:
                G.add_edge(i, j, weight=1)
    return G
#This function generates synthetic samples by randomly selecting nodes from the graph and computing the mean of their features.
#graph: The input graph with nodes containing features.
#num_samples: The number of synthetic samples to generate 
#min_nodes: Minimum number of nodes to select for each sample 
#max_nodes: Maximum number of nodes to select for each sample 
def generate_synthetic_samples(graph, num_samples=1, min_nodes=15, max_nodes=50, seed=None):
    synthetic_samples = []
    if seed is not None:
        random.seed(seed)
    nodes = list(graph.nodes)
    if len(nodes) < min_nodes:
        raise ValueError(f"nodes less than {min_nodes}，cannot generate synthetic samples。")
    for _ in range(num_samples):
        num_nodes = random.randint(min_nodes, min(max_nodes, len(nodes)))
        selected_nodes = random.sample(nodes, k=num_nodes)
        paths = []
        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                path = nx.shortest_path(graph, source=selected_nodes[i], target=selected_nodes[j])
                paths.extend(path)
        unique_nodes = set(paths)
        features = [graph.nodes[node]['feature'] for node in unique_nodes]
        mean_feature = np.mean(features, axis=0)
        synthetic_samples.append(mean_feature)
    return synthetic_samples
#This function processes the input dataset, generates synthetic samples for both positive and negative classes, and writes the generated data to a CSV file.
#file_path: The path to the CSV file containing the dataset.
#num_samples_pos: Number of synthetic samples to generate for the positive class (default is 5).
#num_samples_neg: Number of synthetic samples to generate for the negative class (default is 5).
#seed: The seed for random number generation to ensure reproducibility (default is 2024).

def process_data_and_generate_samples(file_path, num_samples_pos=5, num_samples_neg=5, seed=2024):
    df = pd.read_csv(file_path)
    features_pos = df[df['label'] == 1].iloc[:, 1:].values.tolist()
    features_neg = df[df['label'] == 0].iloc[:, 1:].values.tolist()
    graph_pos = create_full_graph(features_pos)
    graph_neg = create_full_graph(features_neg)
    synthetic_pos = generate_synthetic_samples(graph_pos, num_samples=num_samples_pos, seed=seed)
    synthetic_neg = generate_synthetic_samples(graph_neg, num_samples=num_samples_neg, seed=seed)
    column_names = [f'Feature_{i+1}' for i in range(len(synthetic_pos[0] if synthetic_pos else synthetic_neg[0]))]
    df_pos = pd.DataFrame(synthetic_pos, columns=column_names)
    df_neg = pd.DataFrame(synthetic_neg, columns=column_names)
    df_pos.insert(0, 'label', 1)
    df_neg.insert(0, 'label', 0)
    result_df = pd.concat([df_pos, df_neg], ignore_index=True)
    result_df.to_csv('../DataSet/enhance/prot-t5_gen_equal.csv', index=False)
file_path = '../Dataset/prot-t5_train.csv'
process_data_and_generate_samples(file_path, num_samples_pos=854, num_samples_neg=1, seed=3407)
