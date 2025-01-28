import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to generate datasets
def generate_nonlinear_data(n_variables, n_samples=1000, nonlinear_func=np.exp):
    """
    Generate a dataset where one root variable causes all others using a specified nonlinear function.

    Args:
        n_variables: Number of variables in the dataset.
        n_samples: Number of samples to generate.
        nonlinear_func: Nonlinear function to apply to the root variable.

    Returns:
        data: Pandas DataFrame containing the dataset.
        ground_truth: Ground truth adjacency matrix for the causal structure.
    """
    np.random.seed(42)
    
    # Root cause
    root = np.random.uniform(-1, 1, n_samples)
    
    # Generate dependent variables
    data_dict = {"X1": root}
    for i in range(2, n_variables + 1):
        noise = np.random.uniform(-0.1, 0.1, n_samples)
        data_dict[f"X{i}"] = nonlinear_func(root) + noise
    
    # Ground truth adjacency matrix
    ground_truth = np.zeros((n_variables, n_variables))
    ground_truth[1:, 0] = 1  # Root causes all others
    
    data = pd.DataFrame(data_dict)
    return data, ground_truth

# Function to calculate SHD
def calculate_shd(ground_truth, inferred):
    return np.sum(ground_truth != inferred)

# Function to debug RESIT steps
def debug_resit(model, data, ground_truth):
    logging.info("=== Debugging RESIT ===")
    # Analyze causal order and parent relationships
    causal_order = model.causal_order_
    logging.info(f"Estimated causal order: {causal_order}")
    
    adjacency_matrix = model.adjacency_matrix_
    for i, parents in enumerate(adjacency_matrix):
        variable = f"X{i + 1}"
        parent_indices = np.where(parents == 1)[0]
        parent_names = [f"X{j + 1}" for j in parent_indices]
        
        if not parent_names:
            logging.info(f"Variable {variable} has no parents.")
        else:
            logging.info(f"Variable {variable}:")
            logging.info(f"  Parents: {parent_names}")
    
    # Print results
    logging.info("\nGround Truth Adjacency Matrix:")
    logging.info(f"\n{ground_truth}")
    logging.info("\nInferred Adjacency Matrix:")
    logging.info(f"\n{adjacency_matrix}")

# Function to apply RESIT
def apply_resit(data):
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
    regressor = GaussianProcessRegressor(kernel=kernel, random_state=42)
    model = RESIT(regressor=regressor, alpha=0.05)
    model.fit(data.values)
    return model

# Main Script
if __name__ == "__main__":
    # Generate datasets with 2, 3, and 4 variables
    for n_vars in [2, 3, 4]:
        logging.info(f"\n--- Dataset with {n_vars} Variables ---")
        
        # Generate data
        data, ground_truth = generate_nonlinear_data(n_variables=n_vars, nonlinear_func=np.exp)
        logging.info("Generated Data (First 5 Rows):")
        logging.info(f"\n{data.head()}")
        
        logging.info("\nGround Truth Adjacency Matrix:")
        logging.info(f"\n{ground_truth}")
        
        # Apply RESIT
        model = apply_resit(data)
        
        # Debug RESIT
        debug_resit(model, data.values, ground_truth)
        
        # Calculate SHD
        shd = calculate_shd(ground_truth, model.adjacency_matrix_)
        logging.info(f"\nStructural Hamming Distance (SHD): {shd}")