import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from lingam.hsic import hsic_test_gamma
import logging
import warnings

# Silence all warnings
warnings.filterwarnings("ignore")

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_shd(ground_truth, inferred):
    """Calculate Structural Hamming Distance (SHD)."""
    return np.sum(ground_truth != inferred)


def debug_resit_with_hsic(model, data, ground_truth_dag):
    """Debug RESIT with detailed logs, showing every regression step in Phase 1."""
    logging.info("=== Debugging RESIT ===")
    
    # Phase 1: Topological Ordering
    logging.info("\n=== Phase 1: Topological Ordering ===")
    n_features = data.shape[1]
    remaining_variables = list(range(n_features))
    estimated_order = []
    parent_sets = {i: [] for i in range(n_features)}

    while remaining_variables:
        logging.info("Evaluating sink nodes...")
        best_p_value = float('-inf')
        best_sink = None

        for var in remaining_variables:
            parents = [v for v in remaining_variables if v != var]
            if not parents:
                logging.info(f"Variable X{var + 1}: No parents, assumed independent.")
                p_value = 1.0  # No parents, assume independent
            else:
                model._reg.fit(data[:, parents], data[:, var])
                residuals = data[:, var] - model._reg.predict(data[:, parents])
                hsic_stat, p_value = hsic_test_gamma(residuals, data[:, parents])
                
                # Debug detailed regression results
                logging.info(f"Variable X{var + 1}:")
                logging.info(f"  Parents: {[f'X{p + 1}' for p in parents]}")
                logging.info(f"  Residuals variance: {np.var(residuals):.4f}")
                logging.info(f"  HSIC Stat: {hsic_stat:.4f}, p-value: {p_value:.4f}")

            # Track the best sink node based on the highest p-value
            if p_value > best_p_value:
                best_p_value = p_value
                best_sink = var

        if best_sink is None:
            logging.warning("No valid sink node found. Selecting the variable with the highest p-value.")
            best_sink = max(remaining_variables, key=lambda var: best_p_value)

        # Select the best sink node
        logging.info(f"Selected Sink Node: X{best_sink + 1}, p-value = {best_p_value:.4f}")
        estimated_order.append(best_sink)
        remaining_variables.remove(best_sink)

    # Reverse the estimated order to match topological order
    estimated_order.reverse()
    logging.info(f"Estimated causal order: {estimated_order}")
    logging.info(f"True causal order: {[f'X{i + 1}' for i in range(n_features)]}")
    
    # Create parent sets
    for i, child in enumerate(estimated_order):
        parent_sets[child] = [p for p in estimated_order[:i]]
    logging.info(f"Parent sets for each variable: {parent_sets}")
    
    # Construct adjacency matrix
    adjacency_matrix = np.zeros((n_features, n_features))
    for child, parents in parent_sets.items():
        for parent in parents:
            adjacency_matrix[child, parent] = 1

    # Phase 2: Edge Removal
    logging.info("\n=== Phase 2: Edge Removal ===")
    pa_cleaned = model._remove_edges(data, parent_sets, estimated_order)
    for i, parents in pa_cleaned.items():
        if not parents:
            logging.info(f"Variable X{i + 1}: No parents after edge removal.")
        else:
            logging.info(f"Variable X{i + 1}: Final Parents: {[f'X{p + 1}' for p in parents]}")

    adjacency_matrix_cleaned = np.zeros((n_features, n_features))
    for i, parents in pa_cleaned.items():
        for p in parents:
            adjacency_matrix_cleaned[i, p] = 1

    logging.info(f"Final adjacency matrix:\n{adjacency_matrix_cleaned}")
    return adjacency_matrix_cleaned


def generate_random_dag(n_variables):
    """Generate a random DAG with at least one parent for each variable."""
    dag = np.zeros((n_variables, n_variables), dtype=int)
    for i in range(n_variables):
        for j in range(i):
            if np.random.rand() > 0.5:
                dag[i, j] = 1
        # Ensure at least one parent if not already assigned
        if i > 0 and np.sum(dag[i, :]) == 0:
            dag[i, np.random.randint(0, i)] = 1
    return dag


def generate_nonlinear_data_from_dag(dag, n_samples):
    """Generate nonlinear data based on a given DAG using exponential and Gaussian noise."""
    n_variables = dag.shape[0]
    data = np.zeros((n_samples, n_variables))

    for i in range(n_variables):
        parents = np.where(dag[i, :] == 1)[0]
        if len(parents) == 0:
            data[:, i] = np.random.uniform(-1, 1, n_samples)
        else:
            parent_sum = np.sum(data[:, parents], axis=1)
            data[:, i] = np.exp(parent_sum) + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame(data, columns=[f"X{i+1}" for i in range(n_variables)])


def run_debugging_case(n_variables):
    """Run a single RESIT debugging case."""
    n_samples = 500

    # Generate random DAG and corresponding data
    ground_truth_dag = generate_random_dag(n_variables)
    data = generate_nonlinear_data_from_dag(ground_truth_dag, n_samples)

    # Adjust kernel parameter bounds
    kernel = C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4))
    gaussian_regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model = RESIT(regressor=gaussian_regressor, alpha=0.05)

    inferred_adjacency = debug_resit_with_hsic(model, data.values, ground_truth_dag)
    shd = calculate_shd(ground_truth_dag, inferred_adjacency)

    logging.info("Ground Truth DAG:")
    logging.info(f"\n{ground_truth_dag}")
    logging.info("Inferred Adjacency Matrix:")
    logging.info(f"\n{inferred_adjacency}")
    logging.info(f"SHD: {shd}\n")
    return shd


def run_debugging_for_sizes():
    """Run debugging for sizes 2, 3, and 4 variables."""
    sizes = [2, 3, 4]
    for size in sizes:
        logging.info(f"Starting debugging for {size} variables...")
        shd = run_debugging_case(size)
        logging.info(f"Debugging case for {size} variables completed with SHD={shd}.")


# Run the debugging cases
run_debugging_for_sizes()