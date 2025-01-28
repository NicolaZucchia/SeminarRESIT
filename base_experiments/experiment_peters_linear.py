import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Experiment parameters
random_seed = 42
repetitions = 100
alpha = 0.05
regressor = LinearRegression()

def generate_data(random_state):
    """
    Generate synthetic data based on Peters et al. (2014).

    Args:
        random_state: Random state for reproducibility.

    Returns:
        data: Generated dataset.
        ground_truth: Ground truth adjacency matrix.
    """
    n_samples = 500  # Number of samples
    n_variables = 4  # Number of variables (p = 4)

    # Generate a random sparse DAG
    ground_truth = np.zeros((n_variables, n_variables), dtype=int)
    ordering = random_state.permutation(n_variables)  # Random variable ordering
    for i in range(n_variables):
        for j in range(i):
            if random_state.rand() < 2 / (n_variables - 1):
                ground_truth[ordering[i], ordering[j]] = 1

    # Generate random coefficients for edges in the DAG
    coefficients = np.zeros_like(ground_truth, dtype=float)
    for i in range(n_variables):
        for j in range(n_variables):
            if ground_truth[i, j] == 1:
                coefficients[i, j] = random_state.uniform(-2, -0.1) if random_state.rand() < 0.5 else random_state.uniform(0.1, 2)

    # Generate noise variables
    noise = np.zeros((n_samples, n_variables))
    for j in range(n_variables):
        M = random_state.normal(0, 1, n_samples)  # Mj ~ N(0, 1)
        K = random_state.uniform(0.1, 0.5)  # Kj ~ U([0.1, 0.5])
        alpha = random_state.uniform(2, 4)  # αj ~ U([2, 4])
        noise[:, j] = K * np.sign(M) * np.abs(M) ** alpha  # Non-Gaussian noise

    # Generate data iteratively
    data = np.zeros((n_samples, n_variables))
    for i in range(n_samples):
        for j in range(n_variables):
            parent_indices = np.where(ground_truth[j] == 1)[0]
            data[i, j] = np.sum(coefficients[j, parent_indices] * data[i, parent_indices]) + noise[i, j]

    return data, ground_truth