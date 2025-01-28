import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.linear_model import LinearRegression
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_shd(ground_truth, inferred):
    """
    Calculate Structural Hamming Distance (SHD) between two adjacency matrices.
    """
    return np.sum(ground_truth != inferred)

def custom_hsic_test(X, Y):
    """
    Custom HSIC test based on correlation to estimate independence.
    """
    from scipy.stats import spearmanr
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Y = Y.flatten()
    if X.shape[1] == 1:
        correlation, p_value = spearmanr(X.flatten(), Y)
        return abs(correlation), p_value
    else:
        correlations, p_values = zip(*[spearmanr(X[:, i], Y) for i in range(X.shape[1])])
        return np.mean(correlations), np.mean(p_values)

def debug_resit_with_regression(model, data, noise):
    """
    Debug the RESIT algorithm to inspect regression results and residual independence tests.
    """
    logging.debug("Starting RESIT debugging with regression analysis...")
    pa, pi = model._estimate_order(data)
    logging.debug(f"Estimated causal order: {pi}")
    logging.debug(f"Parent sets for each variable: {pa}")
    for k in pi:
        parents = pa[k]
        if not parents:
            continue
        model._reg.fit(data[:, parents], data[:, k])
        residuals = data[:, k] - model._reg.predict(data[:, parents])
        hsic_stat, hsic_p = custom_hsic_test(data[:, parents], residuals)
        logging.debug(f"Variable X{k+1}, HSIC correlation = {hsic_stat:.4f}, p-value = {hsic_p:.4f}")
    return pa, pi

def generate_random_dag(n_variables):
    """
    Generate a random DAG for the given number of variables.
    """
    dag = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        parents = np.random.choice(range(i), size=np.random.randint(1, i + 1), replace=False)
        dag[i, parents] = 1
    return dag

def generate_data_from_dag(dag, n_samples, noise_variance=0.5):
    """
    Generate synthetic data from the given DAG.
    """
    n_variables = dag.shape[0]
    coefficients = np.random.uniform(0.5, 2.0, size=dag.shape) * dag
    data = np.zeros((n_samples, n_variables))
    for i in range(n_samples):
        for j in range(n_variables):
            parents = np.where(dag[j] == 1)[0]
            parent_data = np.sum(coefficients[j, parents] * data[i, parents]) if len(parents) > 0 else 0
            noise = np.random.uniform(-np.sqrt(3 * noise_variance), np.sqrt(3 * noise_variance))
            data[i, j] = parent_data + noise
    return data

def run_experiment(n_samples, n_variables, repetitions):
    """
    Run multiple experiments with changing ground truth DAGs and data.
    """
    shd_list = []
    for i in range(repetitions):
        logging.info(f"--- Experiment {i+1}/{repetitions} ---")
        ground_truth = generate_random_dag(n_variables)
        logging.debug(f"Ground Truth DAG:\n{ground_truth}")
        data = generate_data_from_dag(ground_truth, n_samples)
        model = RESIT(regressor=LinearRegression(), alpha=0.05)
        model.fit(data)
        inferred_dag = model.adjacency_matrix_
        shd = calculate_shd(ground_truth, inferred_dag)
        shd_list.append(shd)
        logging.info(f"SHD for Experiment {i+1}: {shd}")
    
    mean_shd = np.mean(shd_list)
    std_shd = np.std(shd_list)

    # Print summary of results
    print("\n--- Summary of Results ---")
    print(f"Variables: {n_variables}, SHD Mean ± Std: {mean_shd:.2f} ± {std_shd:.2f}")

    # Print all SHD values for later analysis
    print("\n--- Full SHD Results ---")
    print(f"SHD values: {shd_list}")

if __name__ == "__main__":
    run_experiment(n_samples=500, n_variables=4, repetitions=100)