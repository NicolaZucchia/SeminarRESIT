import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Experiment parameters
random_seed = 42
repetitions = 1
alpha = 0.05
regressor = LinearRegression()

def generate_data(random_state):
    """Generate synthetic linear data"""
    n_samples = 500
    n_variables = 3

    # Ground truth adjacency matrix
    ground_truth = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=float)

    # Coefficients
    coefficients = np.array([
        [0, 0, 0],
        [1.5, 0, 0],
        [0.8, 0, 0],
    ])

    # Generate noise
    noise_variance = 0.5
    noise = random_state.uniform(
        -np.sqrt(3 * noise_variance), np.sqrt(3 * noise_variance), size=(n_samples, n_variables)
    )

    # Generate data
    data = np.zeros((n_samples, n_variables))
    for i in range(n_samples):
        for j in range(n_variables):
            parent_indices = np.where(ground_truth[j] == 1)[0]
            data[i, j] = np.sum(coefficients[j, parent_indices] * data[i, parent_indices]) + noise[i, j]

    return data, ground_truth