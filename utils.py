import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx



def import_facebook_data(data):
    '''Reads the data provided in text format and returns a numpy matrix of shape (n, 2)'''

    df = pd.read_csv(data, sep = ' ', header = None).reset_index(drop = True)
    # int32 format is used to store data based on the largest value of data points (BE CAREFUL OF OVERFLOWS!!)
    nodes_connectivity_list = df.to_numpy(dtype = np.int32)      
    return nodes_connectivity_list 


def import_bitcoin_data(data):
    '''Imports Bitcoin data and does preprocessing on it. Returns a (n, 2) 2d array containing
    the connections between different nodes'''

    df = pd.read_csv(data, sep = ',', header = None).reset_index(drop = True)
    # int32 format is used to store data based on the largest value of data points (BE CAREFUL OF OVERFLOWS!!)
    nodes_connectivity_list = df.to_numpy(dtype = np.int32)[:,:2] - 1   #considering only edges (1st two columns) & converting them to python index system   
    adj_mat_btc = create_adj_mat(nodes_connectivity_list)
    node_degrees = np.sum(adj_mat_btc, 1)
    idx0 = np.array(np.where(node_degrees == 0))        #node_ids of nodes without any edges
    filtered_nodes = {}
    i = 0
    for node in np.arange(np.max(nodes_connectivity_list) + 1):
        if node not in idx0:
            filtered_nodes[node] = i
            i += 1
    updated_nodes_connectivity_list = reset_indices(nodes_connectivity_list, filtered_nodes)    #filtered out nodes without edges
    return updated_nodes_connectivity_list


def create_adj_mat(nodes_connectivity_list):
    '''Creates Adjacency matrix using the nodes connections list (edges)'''

    n = np.max(nodes_connectivity_list)
    adj_mat = np.zeros([n+1,n+1])
    for i in range(nodes_connectivity_list.shape[0]):
        edge1 = nodes_connectivity_list[i,0]
        edge2 = nodes_connectivity_list[i,1]
        adj_mat[edge1, edge2] = 1
        adj_mat[edge2, edge1] = 1
    return adj_mat


def node_to_dict(gpn):
    '''Takes nodes of a graph partition (all nodes belonging to a single community formed
     as a result of partitioning) and returns a dictionary with keys containing the actual nodes
     mapped to integers as per python indexing (0 to n)'''

    nodes = gpn 
    dict = {}
    for i in range(len(nodes)):
        dict[nodes[i]] = i       
    return dict


def get_key(val, my_dict):
    '''Input: Values of the dictionary and the dictionary
    Returns: Keys that map to the values'''

    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def dict_to_node(dict, gpn):
    '''Input: The dictionary and the nodes in python indexing system (0 to n) belonging to a community
    Returns: Actual nodes of the community'''

    for i in range(gpn.shape[0]):
        gpn[i, 0] = get_key(gpn[i,0], dict)
        # gpn[i, 1] = get_key(gpn[i,1], dict)       #can be done in end/ increases time complexity       
    return gpn


def reset_indices(connections, dict):
    '''Input: List containing actual node connections and the dictionary
     that converts these nodes to python indexing
    Returns: List containing the renamed node connections as per python indexing'''

    mapped_connections = []
    for a,b in connections:
        mapped_connections.append([dict[a], dict[b]])
    return np.array(mapped_connections)


def plot_adjMat(adj_mat, resolution = 150):
    '''Plots the Heat Map of the Adjacency Matrix'''

    plt.figure(resolution)
    plt.imshow(adj_mat, cmap = 'twilight_shifted_r')
    plt.title('Adjacency Matrix')
    return plt.show()


def plot_FV(fiedler_vec, resolution = 150):
    '''Plots the Fiedler Vector'''

    nodes = np.arange(np.size(fiedler_vec))
    plt.figure(dpi = resolution)
    plt.scatter(nodes, np.sort(fiedler_vec), s = 0.1)  
    plt.title('Sorted Fiedler Vector')
    return plt.show()


def plot_graph(nodes_connectivity_list, resolution = 150):
    '''Plots the actual graph as per the imported data'''

    plt.figure(dpi = resolution)
    g = nx.Graph()
    g.add_edges_from(nodes_connectivity_list)
    nx.draw(g, node_size = 0.05, node_color = 'red', width = 0.1)
    return plt.show()