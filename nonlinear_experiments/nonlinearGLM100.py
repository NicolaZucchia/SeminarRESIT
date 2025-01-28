import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from lingam.hsic import hsic_test_gamma
import logging
import warnings
import statsmodels.api as sm
from multiprocessing import Pool, cpu_count


# Silence all warnings
warnings.filterwarnings("ignore")

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_shd(ground_truth, inferred):
    """Calculate Structural Hamming Distance (SHD)."""
    return np.sum(ground_truth != inferred)


def debug_resit_with_hsic(model, data, ground_truth_dag):
    """Debug RESIT with detailed logs, showing every regression step in Phase 1."""
    n_features = data.shape[1]
    remaining_variables = list(range(n_features))
    estimated_order = []
    parent_sets = {i: [] for i in range(n_features)}

    while remaining_variables:
        best_p_value = float('-inf')
        best_sink = None

        for var in remaining_variables:
            parents = [v for v in remaining_variables if v != var]
            if not parents:
                p_value = 1.0  # No parents, assume independent
            else:
                X_parents = data[:, parents]
                y = data[:, var]
                X_parents = np.maximum(X_parents, 1e-10)  # Replace non-positive values
                y = np.maximum(y, 1e-10)  # Replace non-positive values

                # Add constant term for GLM regression
                X_parents = sm.add_constant(X_parents)
                
                # Fit GLM with exponential family
                glm = sm.GLM(y, X_parents, family=sm.families.Gamma(link=sm.families.links.log()))
                glm_result = glm.fit()

                # Calculate residuals
                residuals = y - glm_result.predict(X_parents)

                # Perform HSIC test
                hsic_stat, p_value = hsic_test_gamma(residuals, X_parents)

            # Track the best sink node based on the highest p-value
            if p_value > best_p_value:
                best_p_value = p_value
                best_sink = var

        estimated_order.append(best_sink)
        remaining_variables.remove(best_sink)

    # Reverse the estimated order to match topological order
    estimated_order.reverse()

    # Create parent sets
    for i, child in enumerate(estimated_order):
        parent_sets[child] = [p for p in estimated_order[:i]]
    
    # Construct adjacency matrix
    adjacency_matrix = np.zeros((n_features, n_features))
    for child, parents in parent_sets.items():
        for parent in parents:
            adjacency_matrix[child, parent] = 1

    return adjacency_matrix


def generate_random_dag(n_variables):
    """Generate a random DAG with at least one parent for each variable."""
    dag = np.zeros((n_variables, n_variables), dtype=int)
    for i in range(n_variables):
        for j in range(i):
            if np.random.rand() > 0.5:
                dag[i, j] = 1
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
            data[:, i] = np.random.uniform(0.1, 1, n_samples)  # Ensure strictly positive values
        else:
            parent_sum = np.sum(data[:, parents], axis=1)
            data[:, i] = np.exp(parent_sum) + np.abs(np.random.normal(0, 0.1, n_samples))  # Enforce positivity
    return pd.DataFrame(data, columns=[f"X{i+1}" for i in range(n_variables)])


def run_single_experiment(n_variables):
    """Run a single RESIT debugging case."""
    n_samples = 500
    ground_truth_dag = generate_random_dag(n_variables)
    data = generate_nonlinear_data_from_dag(ground_truth_dag, n_samples)
    kernel = C(1.0, (1e-2, 1e4)) * RBF(1.0, (1e-2, 1e4))
    gaussian_regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model = RESIT(regressor=gaussian_regressor, alpha=0.05)
    inferred_adjacency = debug_resit_with_hsic(model, data.values, ground_truth_dag)
    shd = calculate_shd(ground_truth_dag, inferred_adjacency)
    return shd


def run_parallel_experiments():
    """Run 100 experiments in parallel for sizes 2, 3, and 4 variables."""
    sizes = [2, 3, 4]
    num_experiments = 100
    results = {size: [] for size in sizes}
    
    with Pool(processes=cpu_count()) as pool:
        for size in sizes:
            logging.info(f"Starting parallel experiments for {size} variables...")
            shd_results = pool.map(run_single_experiment, [size] * num_experiments)
            results[size] = shd_results
            logging.info(f"Completed parallel experiments for {size} variables.")
    
    for size, shds in results.items():
        logging.info(f"Results for {size} variables:")
        logging.info(f"Mean SHD: {np.mean(shds):.2f}, Std Dev: {np.std(shds):.2f}, Min: {np.min(shds)}, Max: {np.max(shds)}")


if __name__ == '__main__':
    # Run the parallel experiments
    run_parallel_experiments()