import numpy as np
from utils import node_to_dict, dict_to_node, reset_indices


###############################################################################################################
###############################################################################################################


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


def create_degree_mat(adj_mat):
    '''Creates the Degree Matrix, where the diagonal elements give the degree of the corresponding node'''

    node_degree = np.sum(adj_mat, 1)
    D = np.diag(node_degree)
    return D


def compute_fiedler_vec(L, D):
    '''Computes Fiedler vector using the Graph Laplacian and the Degree matrix'''

    D_inv_sqrt = np.sqrt(np.linalg.pinv(D))
    L_D_inv_sqrt = np.matmul(L, D_inv_sqrt)
    std_eig_sys = np.matmul(D_inv_sqrt, L_D_inv_sqrt)   #performs normalized cut (solves Lx = λDx)
    eig_vals, eig_vecs = np.linalg.eig(std_eig_sys)     # gives right eigenvectors i.e., eigenvectors arranged in columns
    threshold = 0.000000001

    sorted_eig_vals = np.sort(eig_vals)
    boolean =   sorted_eig_vals > threshold
    for idx in range(len(boolean)):
        if boolean[idx] == 1:
            break

    eig_val = sorted_eig_vals[idx]
    idx_fiedler_vec = np.where(eig_vals == eig_val)
    fiedler_vec = eig_vecs.T[idx_fiedler_vec]           #transposed eigenvector matrix
    return fiedler_vec


def create_edges(gpn, adj_mat):                     # gpn = graph_partition_nodes (o/p of spectralDecomp_OneIter)
    '''Creates edges between nodes using the Graph nodes and the original Adjacency matrix formed by Initial data'''

    nodes = gpn                                                # community member nodes
    edges = []
    for node in nodes:
        node_edges = np.where(adj_mat[node, node:] == 1)       # checks edges above the principal diagonal
        node_edges = np.array(node_edges) + node               # corrects indices
        for i in range(node_edges.shape[1]):
            if np.isin(node_edges[0, i], nodes):               # checks if edges are within the same community
                edges.append([node, node_edges[0, i]])
    edges = np.array(edges)
    return edges


def createSortedAdjMat(graph_partition, nodes_connectivity_list): 
    '''Creates sorted Adjacency matrix sorted by increasing order of community ids,
     where id of a community is the smallest node id present in the community'''

    adj_mat = create_adj_mat(nodes_connectivity_list)
    sorted_indices = np.argsort(graph_partition[:, 1], kind = 'mergesort')
    sorted_gp = graph_partition[sorted_indices]
    n = len(sorted_gp)
    i = 0
    sorted_adj = np.zeros([n+1,n+1])
    for row in sorted_gp[:,0]:
        j = 0
        for col in sorted_gp[:,0]:
            sorted_adj[i,j] = adj_mat[row, col]
            j+=1
        i+=1
    return sorted_adj


###########################################################################################
###########################################################################################
################################ One Iteration of Spectral Decomposition ##################
###########################################################################################
###########################################################################################


def spectralDecomp_OneIter(nodes_connectivity_list):
    '''Runs one iteration of the Spectral Decomposition Algorithm'''

    # Array initialization
    n = np.max(nodes_connectivity_list)
    adj_mat = np.zeros([n+1,n+1])
    D = np.zeros([n+1,n+1])
    graph_partition = np.zeros([n+1, 2])

    # Adjacency matrix
    adj_mat = create_adj_mat(nodes_connectivity_list)

    # Degree Matrix
    D = create_degree_mat(adj_mat)

    # Graph Laplacian
    L = D - adj_mat

    # Fiedler vector computation
    # eig_val, eig_vec = scipy.sparse.linalg.eigs(L, k = 2, M = D, which = 'SM')
    # fiedler_vec = eig_vec[:,1]
    fiedler_vec = compute_fiedler_vec(L, D)   

    # Graph Partition
    nodes = np.arange(n+1)
    _, community1 = np.where(fiedler_vec > 0)
    _, community2 = np.where(fiedler_vec < 0)
    graph_partition[:,0] = nodes
    graph_partition[community1, 1] = np.min(community1)
    graph_partition[community2, 1] = np.min(community2)

    return fiedler_vec, adj_mat, graph_partition.astype(np.int32)



###########################################################################################
###########################################################################################
################################ Spectral Decomposition Algorithm #########################
###########################################################################################
###########################################################################################


def spectralDecomposition(nodes_connectivity_list, community_size_threshold = 300):
    '''Runs the Spectral Decomposition Algorithm until community size threshold 
    is breached for all communities formed'''

    max_iter = 500
    ncl = nodes_connectivity_list
    n = np.max(nodes_connectivity_list)
    partition = {}
    final_partition = {}
    bool = True
    k = 0
    j = 0
    l = 0

    for i in range(max_iter):       
        fv, am, gp = spectralDecomp_OneIter(ncl)

        if i == 0:
            adj_mat = am

        if i > 0:
            gp = dict_to_node(dict, gp)
    
        if bool:
            partition['gp' + str(k)] = gp[ np.where( gp[:,1] == np.min(gp[:,1]) ), 0].flatten()
            k += 1

            partition['gp' + str(k)] = gp[ np.where( gp[:,1] == np.max(gp[:,1]) ), 0].flatten()
            k += 1

        if not bool:
            final_partition['community_' + str(l)] = partition['gp' + str(j-1)]
            l += 1

        bool = True
        
        if j == k:
            break

        edges = create_edges(partition['gp' + str(j)], adj_mat)
        dict = node_to_dict(partition['gp' + str(j)])
        ncl = reset_indices(edges, dict)
               
        if len(partition['gp' + str(j)]) < community_size_threshold:
            bool = False

        j += 1
     
    graph_partition = np.empty((0,2), int)
    for i in range(l):
         gpc0 = final_partition['community_' + str(i)]
         gpc0 = gpc0.reshape([-1, 1])
         gpc1 = np.min(final_partition['community_' + str(i)]) * np.ones(len(gpc0))
         gpc1 = gpc1.reshape([-1, 1])
         gp_hcat = np.hstack((gpc0, gpc1))
         graph_partition = np.vstack((graph_partition, gp_hcat))

    return graph_partition.astype(np.int32)