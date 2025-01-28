import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Experiment parameters
random_seed = 42
repetitions = 1
alpha = 0.05
regressor = GaussianProcessRegressor(kernel=RBF(), alpha=0.05)  # Nonlinear regression using Gaussian process

def generate_data(random_state):
    """Generate synthetic nonlinear data"""
    n_samples = 500
    n_variables = 2  # Only two variables

    # Ground truth adjacency matrix
    ground_truth = np.array([
        [0, 0],  # X1 has no parents
        [1, 0],  # X2 depends on X1
    ], dtype=float)

    # Generate noise
    noise_variance = 0.5
    noise = random_state.normal(0, np.sqrt(noise_variance), size=(n_samples, n_variables))

    # Generate data
    data = np.zeros((n_samples, n_variables))
    data[:, 0] = random_state.uniform(-1, 1, size=n_samples)  # X1 uniformly distributed
    data[:, 1] = data[:, 0] ** 2 + noise[:, 1]  # X2 = X1^2 + noise

    return data, ground_truth