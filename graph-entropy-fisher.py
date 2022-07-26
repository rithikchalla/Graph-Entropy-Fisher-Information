import numpy as np
from math import sqrt

def graph_entropy(adjacency_matrix):
    """
        Function is used to compute normalized graph entropy and normalized graph fisher information from the
        adjacency matrix of the graph

        param adjacency_matrix: A square numpy 2D array
        return: a tuple of normalized graph entropy and normalized graph fisher information
    """
    
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise Exception("Input matrix should be a square matrix")
    
    # This will return the number of nodes in the graph
    nodes = adjacency_matrix.shape[0]
    
    # Computes graph entropy from (Equation 4 from reference article)
    normalized_graph_entropy = (1/(nodes * np.log(nodes-1))) * (sum(np.log(np.sum(adjacency_matrix, axis=0))))

    # Generate random walker path from the adjacency matrix (Equation 1 from the reference article)
    random_walk_mat = np.zeros(adjacency_matrix.shape)
    for i in range(nodes):
        random_walk_mat[i,] = adjacency_matrix[i,]/np.sum(adjacency_matrix[i,])

    temp = np.zeros((1, nodes))
    node_information = np.zeros((1, nodes))

    # Compute normalized graph fisher information (Equation 5 and Equation 6 from the reference article)
    for k in range(nodes):
        list_values = list(map((lambda l: ((sqrt(random_walk_mat[k, l+1]) - sqrt(random_walk_mat[k, l])) ** 2) if k != l else 0), range(nodes-1)))
        node_information[0, k] = (0.5 * (sum(list_values)))

    normalized_graph_fisher_information = np.sum(node_information[0,], axis=0)/nodes

    return (normalized_graph_entropy, normalized_graph_fisher_information)