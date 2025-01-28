import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.linear_model import LinearRegression
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_data(n_samples, n_variables, seed=None):
    """
    Generate data according to the settings in Section 5.1.1 of Peters et al. (2014).
    """
    np.random.seed(seed)
    adjacency_matrix = np.zeros((n_variables, n_variables))
    coefficients = np.zeros((n_variables, n_variables))
    data = np.zeros((n_samples, n_variables))

    # Create random DAG
    for i in range(n_variables):
        for j in range(i):
            if np.random.rand() < 0.5:  # 50% chance of having an edge
                adjacency_matrix[i, j] = 1
                coefficients[i, j] = np.random.choice([-1, -0.5, 0.5, 1])

    # Generate data
    noise = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(n_samples, n_variables))
    for i in range(n_variables):
        parent_indices = np.where(adjacency_matrix[i] == 1)[0]
        for j in range(n_samples):
            data[j, i] = np.sum(coefficients[i, parent_indices] * data[j, parent_indices]) + noise[j, i]

    return pd.DataFrame(data, columns=[f"X{k + 1}" for k in range(n_variables)]), adjacency_matrix

def run_experiment(n_samples, n_variables, seed=None):
    """
    Run RESIT on generated data with given n_samples and n_variables.
    """
    data, ground_truth = generate_data(n_samples, n_variables, seed=seed)
    logging.info(f"Generated data with n_samples={n_samples}, n_variables={n_variables}")

    # Apply RESIT
    model = RESIT(regressor=LinearRegression(), alpha=0.05)

    # Fit the model
    model.fit(data.values)

    # Evaluate results
    inferred_adjacency = model.adjacency_matrix_
    shd = np.sum(ground_truth != inferred_adjacency)
    return shd

def main():
    """
    Main experiment loop for p = 3 to 10 with n = 500, repeated 100 times.
    """
    n_samples = 500
    n_variables_list = range(3, 11)
    repetitions = 100

    summary_results = []

    for n_variables in n_variables_list:
        logging.info(f"Running experiments for n_variables={n_variables}")
        shd_results = []
        for iteration in range(1, repetitions + 1):
            logging.info(f"Iteration {iteration} for n_variables={n_variables}")
            shd = run_experiment(n_samples, n_variables, seed=iteration)
            shd_results.append(shd)
        
        mean_shd = np.mean(shd_results)
        stddev_shd = np.std(shd_results)
        summary_results.append((n_variables, mean_shd, stddev_shd))
        logging.info(f"Results for n_variables={n_variables}: SHD = {mean_shd:.2f} ± {stddev_shd:.2f}")

    # Print summary
    print("\nSummary of Results:")
    print("Variables (p) | Mean SHD ± StdDev")
    print("-" * 30)
    for n_variables, mean_shd, stddev_shd in summary_results:
        print(f"{n_variables:>12} | {mean_shd:.2f} ± {stddev_shd:.2f}")

if __name__ == "__main__":
    main()