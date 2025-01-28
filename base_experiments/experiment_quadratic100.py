import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from lingam.hsic import hsic_test_gamma
import logging
from joblib import Parallel, delayed

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_shd(ground_truth, inferred):
    """Calculate Structural Hamming Distance (SHD)."""
    return np.sum(ground_truth != inferred)

def debug_resit_with_hsic(model, data):
    """Debug RESIT with HSIC independence test."""
    logging.debug("Starting RESIT debugging with HSIC analysis...")
    pa, pi = model._estimate_order(data)
    logging.debug(f"Estimated causal order: {pi}")
    logging.debug(f"Parent sets for each variable: {pa}")
    
    for k in pi:
        parents = pa[k]
        if not parents:
            continue
        model._reg.fit(data[:, parents], data[:, k])
        predicted = model._reg.predict(data[:, parents])
        residuals = data[:, k] - predicted
        hsic_stat, hsic_p = hsic_test_gamma(residuals, data[:, parents])
        logging.debug(f"Variable X{k+1}, HSIC p-value: {hsic_p:.4f}")

    pa_cleaned = model._remove_edges(data, pa, pi)
    n_features = data.shape[1]
    adjacency_matrix = np.zeros([n_features, n_features])
    for i, parents in pa_cleaned.items():
        for p in parents:
            adjacency_matrix[i, p] = 1

    logging.debug(f"Final adjacency matrix:\n{adjacency_matrix}")
    return pa, pi, adjacency_matrix

def generate_random_dag(n_variables):
    """Generate a random DAG with edges below the diagonal."""
    dag = np.zeros((n_variables, n_variables), dtype=int)
    for i in range(n_variables):
        for j in range(i):
            if np.random.rand() > 0.5:  # Randomly generate edges with 50% probability
                dag[i, j] = 1
    return dag

def generate_nonlinear_data_from_dag(dag, n_samples):
    """Generate nonlinear data based on a given DAG using squared relationships and uniform noise."""
    n_variables = dag.shape[0]
    data = np.zeros((n_samples, n_variables))

    for i in range(n_variables):
        parents = np.where(dag[i, :] == 1)[0]
        if len(parents) == 0:
            data[:, i] = np.random.uniform(-1, 1, n_samples)  # Root node
        else:
            parent_sum = np.sum(data[:, parents] ** 2, axis=1)  # Squared relationships
            data[:, i] = parent_sum + np.random.uniform(-0.1, 0.1, n_samples)  # Add uniform noise
    
    return pd.DataFrame(data, columns=[f"X{i+1}" for i in range(n_variables)])

def run_single_experiment(run_id):
    """Run a single RESIT experiment and return SHD with ID."""
    n_samples = 500
    n_variables = 4  # Change if needed

    # Generate random DAG and corresponding data
    ground_truth_dag = generate_random_dag(n_variables)
    data = generate_nonlinear_data_from_dag(ground_truth_dag, n_samples)

    kernel = C(1.0, (1e-2, 1e3)) * RBF(1.0, (1e-2, 1e3))
    gaussian_regressor = GaussianProcessRegressor(kernel=kernel, random_state=0)
    model = RESIT(regressor=gaussian_regressor, alpha=0.05)

    _, _, adjacency_matrix = debug_resit_with_hsic(model, data.values)
    model.fit(data.values)
    inferred_adjacency = model.adjacency_matrix_
    shd = calculate_shd(ground_truth_dag, inferred_adjacency)

    logging.info(f"Run {run_id + 1}: SHD = {shd}")  # Print each SHD result
    return shd

def run_experiment_with_hsic_parallel():
    """Run multiple experiments in parallel and print all SHD values."""
    n_runs = 100  # Number of parallel runs
    results = Parallel(n_jobs=-1)(delayed(run_single_experiment)(i) for i in range(n_runs))
    
    logging.info(f"\nAll SHD values: {results}")
    logging.info(f"Average SHD: {np.mean(results):.4f}")
    logging.info(f"Standard Deviation of SHD: {np.std(results):.4f}")

run_experiment_with_hsic_parallel()