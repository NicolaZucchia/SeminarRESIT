import numpy as np
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import matplotlib.pyplot as plt

def parse_dag(dag_string):
    """
    Parse a printed DAG string into a NumPy adjacency matrix.

    Args:
        dag_string (str): A string representation of the DAG, e.g.:
                          [[0. 0. 0.]
                           [1. 0. 0.]
                           [1. 0. 0.]]
    Returns:
        numpy.ndarray: Parsed adjacency matrix.
    """
    rows = dag_string.strip().replace('[', '').replace(']', '').split('\n')
    matrix = [list(map(float, row.split())) for row in rows]
    return np.array(matrix)

def plot_hierarchical_dag(dag_string, filename="hierarchical_dag.png"):
    """
    Plot a hierarchical DAG from a string representation of its adjacency matrix and save it to an image file.

    Args:
        dag_string (str): A string representation of the adjacency matrix.
        filename (str, optional): Name of the file to save the image. Defaults to "hierarchical_dag.png".
    """
    adjacency_matrix = parse_dag(dag_string)

    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    n_nodes = adjacency_matrix.shape[0]
    node_labels = [f"X{i+1}" for i in range(n_nodes)]

    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes and edges based on the adjacency matrix
    for i in range(n_nodes):
        G.add_node(i, label=node_labels[i])
        for j in range(n_nodes):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(j, i)  # Flip direction: j causes i

    # Use Graphviz layout for a hierarchical structure
    pos = graphviz_layout(G, prog="dot")

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={i: label for i, label in enumerate(node_labels)},
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        arrowsize=15,
    )

    # Save the image
    plt.savefig(filename, format="png")
    plt.close()
    print(f"DAG saved as {filename}")

# Example usage
if __name__ == "__main__":
    dag_string = """
    [[0. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]]
    """
    plot_hierarchical_dag(dag_string, filename="DAG.png")