import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.linear_model import LinearRegression
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to calculate SHD
def calculate_shd(ground_truth, inferred):
    """Calculate Structural Hamming Distance (SHD) between two adjacency matrices."""
    return np.sum(ground_truth != inferred)

# Function to generate random DAG
def generate_random_dag(n_variables):
    """Generate a random DAG with n_variables."""
    dag = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        for j in range(i):
            if np.random.rand() < 0.5:  # Randomly decide if there is an edge
                dag[i, j] = 1
    return dag

# Function to generate synthetic data from a DAG
def generate_data_from_dag(dag, n_samples, noise_type="uniform", noise_variance=0.5):
    """Generate data based on a given DAG."""
    n_variables = dag.shape[0]
    data = np.zeros((n_samples, n_variables))
    for i in range(n_variables):
        parents = np.where(dag[i, :] == 1)[0]
        if parents.size == 0:
            data[:, i] = np.random.uniform(-1, 1, n_samples)
        else:
            noise = (
                np.random.normal(0, np.sqrt(noise_variance), n_samples)
                if noise_type == "gaussian"
                else np.random.uniform(-np.sqrt(3 * noise_variance), np.sqrt(3 * noise_variance), n_samples)
            )
            data[:, i] = np.sum(data[:, parents], axis=1) + noise
    return pd.DataFrame(data, columns=[f"X{i+1}" for i in range(n_variables)])

# Function to debug RESIT and print detailed information
def debug_resit(model, data, ground_truth):
    """Debug RESIT algorithm by analyzing regression results and adjacency matrix."""
    logging.info("\n=== Debugging RESIT ===")
    adjacency_matrix = model.adjacency_matrix_
    shd = calculate_shd(ground_truth, adjacency_matrix)
    
    # Print causal order
    causal_order = model.causal_order_
    logging.info(f"Estimated causal order: {causal_order}")
    
    logging.info("\n=== Results ===")
    logging.info(f"Ground Truth Adjacency Matrix:\n{ground_truth}")
    logging.info(f"Inferred Adjacency Matrix:\n{adjacency_matrix}")
    logging.info(f"Structural Hamming Distance (SHD): {shd}")
    return shd

# Main experiment function
def run_experiment():
    n_samples = 500
    n_variables = 5
    noise_variance = 0.5
    n_runs = 100
    
    results_uniform = []
    results_gaussian = []
    
    for run in range(n_runs):
        logging.info(f"\n--- Experiment Run {run + 1} ---")
        dag = generate_random_dag(n_variables)
        
        # Uniform noise
        logging.info("\n--- Uniform Noise ---")
        data_uniform = generate_data_from_dag(dag, n_samples, noise_type="uniform", noise_variance=noise_variance)
        model_uniform = RESIT(regressor=LinearRegression(), alpha=0.05)
        model_uniform.fit(data_uniform.values)
        shd_uniform = debug_resit(model_uniform, data_uniform.values, dag)
        results_uniform.append(shd_uniform)
        
        # Gaussian noise
        logging.info("\n--- Gaussian Noise ---")
        data_gaussian = generate_data_from_dag(dag, n_samples, noise_type="gaussian", noise_variance=noise_variance)
        model_gaussian = RESIT(regressor=LinearRegression(), alpha=0.05)
        model_gaussian.fit(data_gaussian.values)
        shd_gaussian = debug_resit(model_gaussian, data_gaussian.values, dag)
        results_gaussian.append(shd_gaussian)
    
    # Summarize results
    logging.info("\n=== Summary of Results ===")
    logging.info(f"Uniform Noise - Mean SHD: {np.mean(results_uniform):.4f}, Std Dev: {np.std(results_uniform):.4f}")
    logging.info(f"Gaussian Noise - Mean SHD: {np.mean(results_gaussian):.4f}, Std Dev: {np.std(results_gaussian):.4f}")

    # Print all SHD values for later visualization
    logging.info("\n=== Full SHD Results ===")
    logging.info(f"Uniform Noise SHD values: {results_uniform}")
    logging.info(f"Gaussian Noise SHD values: {results_gaussian}")

# Run the experiment
if __name__ == "__main__":
    run_experiment()