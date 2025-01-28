import importlib.util
import os
import logging
from sklearn.utils import check_random_state
import numpy as np
# Common utilities
from lingam.resit import RESIT
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from lingam.hsic import hsic_test_gamma

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Enable logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import random
random.seed = 42

def calculate_shd(ground_truth, inferred):
    """Calculate Structural Hamming Distance (SHD)"""
    return int((ground_truth != inferred).sum())

def debug_resit_with_hsic(model, data, ground_truth):
    """Debugging RESIT with detailed logs."""
    logging.info("=== Debugging RESIT ===")

    # Phase 1: Topological Ordering
    logging.info("\n=== Phase 1: Topological Ordering ===")
    pa, pi = model._estimate_order(data)
    n_features = data.shape[1]

    # Debug p-values during causal order estimation
    estimated_order = []
    remaining_variables = list(range(n_features))
    while remaining_variables:
        best_p_value = float('-inf')
        best_sink = None

        for var in remaining_variables:
            parents = [v for v in remaining_variables if v != var]
            if not parents:
                residuals = data[:, var]
                p_value = 1.0  # No parents, assume independence
            else:
                model._reg.fit(data[:, parents], data[:, var])
                residuals = data[:, var] - model._reg.predict(data[:, parents])
                hsic_stat, p_value = hsic_test_gamma(residuals, data[:, parents])
            
            logging.debug(f"Variable X{var + 1}: p-value = {p_value:.4f}")

            if p_value > best_p_value:
                best_p_value = p_value
                best_sink = var

        if best_sink is None:
            logging.error("No valid sink found; aborting phase 1.")
            return None, None

        estimated_order.append(best_sink)
        remaining_variables.remove(best_sink)
        logging.info(f"Selected Sink: X{best_sink + 1}, p-value = {best_p_value:.4f}")

    # Reverse the estimated order to match topological order
    estimated_order.reverse()

    logging.info(f"Estimated causal order: {estimated_order}")
    logging.info(f"True causal order: {list(range(n_features))}")
    rank_corr = np.corrcoef(list(range(n_features)), estimated_order)[0, 1]
    logging.info(f"Rank Correlation: {rank_corr:.4f}")

    # Full DAG after Phase 1
    full_dag = np.zeros((n_features, n_features))
    for i, parents in pa.items():
        for p in parents:
            full_dag[i, p] = 1
    logging.info(f"\nFull DAG after Phase 1 (Topological Ordering):\n{full_dag}")

    # Phase 2: Edge Removal
    logging.info("\n=== Phase 2: Edge Removal ===")
    for k, target in enumerate(pi):
        parents = pa[target]
        if not parents:
            logging.info(f"Variable X{target + 1} has no parents.")
            continue

        logging.info(f"Variable X{target + 1}:")
        logging.info(f"  Initial Parents: {[f'X{p + 1}' for p in parents]}")

        for parent in parents[:]:
            current_parents = [p for p in parents if p != parent]
            if not current_parents:
                logging.info(f"  Edge X{parent + 1} -> X{target + 1} not checked (no other parents).")
                continue

            model._reg.fit(data[:, current_parents], data[:, target])
            residuals = data[:, target] - model._reg.predict(data[:, current_parents])
            hsic_stat, hsic_p = hsic_test_gamma(residuals, data[:, parent])
            logging.info(f"  Checking Edge X{parent + 1} -> X{target + 1}:")
            logging.info(f"    HSIC Stat: {hsic_stat:.4f}, p-value: {hsic_p:.4f}")

            if hsic_p > model._alpha:
                logging.info(f"    Edge X{parent + 1} -> X{target + 1} pruned (p > {model._alpha}).")
                parents.remove(parent)
            else:
                logging.info(f"    Edge X{parent + 1} -> X{target + 1} kept (p <= {model._alpha}).")

        logging.info(f"  Final Parents: {[f'X{p + 1}' for p in parents]}")

    # Final adjacency matrix after Phase 2
    inferred_adjacency = np.zeros((n_features, n_features))
    for i, parents in pa.items():
        for p in parents:
            inferred_adjacency[i, p] = 1

    # Results
    shd = calculate_shd(ground_truth, inferred_adjacency)
    logging.info("\n=== Results ===")
    logging.info(f"Ground Truth Adjacency Matrix:\n{ground_truth}")
    logging.info(f"Inferred Adjacency Matrix:\n{inferred_adjacency}")
    logging.info(f"Structural Hamming Distance (SHD): {shd}")

    return shd


def run_experiment(experiment_file):
    """Run a single experiment."""
    spec = importlib.util.spec_from_file_location("experiment", experiment_file)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)

    random_state = check_random_state(experiment.random_seed)
    repetitions = experiment.repetitions
    shd_results = []  # Store SHD values for each repetition

    for i in range(repetitions):
        logging.info(f"\n=== Repetition {i + 1}/{repetitions} ===")
        data, ground_truth = experiment.generate_data(random_state)

        model = RESIT(regressor=experiment.regressor, alpha=experiment.alpha)
        model.fit(data)

        shd = debug_resit_with_hsic(model, data, ground_truth)
        shd_results.append(shd)  # Save SHD for this iteration

    # Summary
    mean_shd = sum(shd_results) / len(shd_results)
    std_dev_shd = np.std(shd_results)

    # Log summary
    logging.info(f"\n=== Experiment Summary ===")
    logging.info(f"Mean SHD over {repetitions} repetitions: {mean_shd:.2f}")
    logging.info(f"Standard Deviation of SHD over {repetitions} repetitions: {std_dev_shd:.2f}")

    # Display all SHD values
    logging.info("\n=== Full SHD Results ===")
    logging.info(f"SHD values for all iterations: {shd_results}")

    print("\n--- Summary of Results ---")
    print(f"Mean SHD: {mean_shd:.2f}")
    print(f"Std Dev SHD: {std_dev_shd:.2f}")
    print(f"SHD values: {shd_results}")

if __name__ == "__main__":
    experiment_script = "experiment_quadratic.py"  # Replace with your script path
    run_experiment(experiment_script)