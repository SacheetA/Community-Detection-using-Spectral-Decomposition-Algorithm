from utils import import_facebook_data, import_bitcoin_data, plot_FV, plot_adjMat, plot_graph
from spectral_decomposition import spectralDecomp_OneIter, spectralDecomposition, createSortedAdjMat


if __name__ == "__main__":

    ####################### FaceBook Dataset (Code may take 4-5 mins to run) ######################
    ''' Import facebook_combined.txt
     nodes_connectivity_list is a nx2 numpy array, where every row 
     is a edge connecting i<->j (entry in the first column is node i, 
     entry in the second column is node j)
     Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.'''
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    plot_graph(nodes_connectivity_list_fb, resolution = 200)

     '''fielder_vec    : n-length numpy array. (n being number of nodes in the network)
     adj_mat        : nxn adjacency matrix of the graph
     graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
                      nodes in the network and the second column lists their community id (starting from 0)
                      Follow the convention that the community id is equal to the lowest nodeID in that community.'''
    fiedler_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    plot_FV(fiedler_vec_fb, resolution = 200)
    plot_adjMat(adj_mat_fb, resolution = 200)
    
     '''graph_partition is a nx2 numpy array, as before. It now contains all the community id's that have been
     identified. The naming convention for the community id is as before.'''
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb, community_size_threshold = 450)

     '''Create the sorted adjacency matrix of the entire graph. The
     adjacency matrix is to be sorted in an increasing order of communitites.'''
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    #sorted adjacency matrix created after forming the communites
    plot_adjMat(clustered_adj_mat_fb)


    
    
    ################# For Bitcoin Dataset Uncomment The Following Lines (Code may take 10-15 min to run based on system configuration) ######################
    '''Note that nodes without any edges have been filtered out, the import function can be altered 
    to plot the actual dataset, but while running the algorithm it is advisable to filter out individual 
    nodes without any edges.'''
    
    # nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")
    # plot_graph(nodes_connectivity_list_btc, resolution = 200)  
    # fiedler_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
    # plot_FV(fiedler_vec_btc, resolution = 200)
    # plot_adjMat(adj_mat_btc, resolution = 200)
    # graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)
    # clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)
    # plot_adjMat(clustered_adj_mat_btc)
