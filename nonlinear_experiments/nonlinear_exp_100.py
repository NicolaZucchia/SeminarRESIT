import warnings
from itertools import product
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to generate datasets
def generate_nonlinear_data(n_variables, n_samples=1000, nonlinear_func=np.exp):
    np.random.seed()  # Ensure different random seeds for each experiment
    root = np.random.uniform(-1, 1, n_samples)
    data_dict = {"X1": root}
    for i in range(2, n_variables + 1):
        noise = np.random.uniform(-0.1, 0.1, n_samples)
        data_dict[f"X{i}"] = nonlinear_func(root) + noise
    ground_truth = np.zeros((n_variables, n_variables))
    ground_truth[1:, 0] = 1
    data = pd.DataFrame(data_dict)
    return data, ground_truth

# Function to calculate SHD
def calculate_shd(ground_truth, inferred):
    return np.sum(ground_truth != inferred)

# Function to apply RESIT
def apply_resit(data, constant_bound, length_scale_bound):
    kernel = ConstantKernel(1.0, constant_bound) * RBF(1.0, length_scale_bound)
    regressor = GaussianProcessRegressor(kernel=kernel, random_state=42)
    model = RESIT(regressor=regressor, alpha=0.05)
    model.fit(data.values)
    return model

# Function to run experiments
def run_experiments(n_trials=100, constant_bounds=(1e-2, 1e2), length_scale_bounds=(1e-2, 1e2)):
    results = {2: [], 3: [], 4: []}  # Store SHD for each variable size

    for n_vars in [2, 3, 4]:
        logging.info(f"\n--- Running {n_trials} Experiments for {n_vars} Variables ---")
        shd_list = []
        for trial in range(n_trials):
            # Generate data
            data, ground_truth = generate_nonlinear_data(n_variables=n_vars)
            
            # Apply RESIT
            model = apply_resit(data, constant_bounds, length_scale_bounds)
            
            # Calculate SHD
            shd = calculate_shd(ground_truth, model.adjacency_matrix_)
            shd_list.append(shd)
        
        # Store results
        results[n_vars] = shd_list
        mean_shd = np.mean(shd_list)
        std_shd = np.std(shd_list)
        logging.info(f"Average SHD for {n_vars} variables over {n_trials} trials: {mean_shd:.4f}")
        logging.info(f"SHD Standard Deviation for {n_vars} variables: {std_shd:.4f}")
        logging.info(f"SHD distribution: {shd_list}")

    return results

# Main Script
if __name__ == "__main__":
    n_trials = 100  # Number of repetitions for the experiment
    results = run_experiments(n_trials=n_trials)
    
    # Summarize results
    for n_vars, shd_list in results.items():
        mean_shd = np.mean(shd_list)
        std_shd = np.std(shd_list)
        logging.info(f"\nSummary for {n_vars} variables over {n_trials} trials:")
        logging.info(f"  Mean SHD: {mean_shd:.4f}")
        logging.info(f"  Standard Deviation: {std_shd:.4f}")