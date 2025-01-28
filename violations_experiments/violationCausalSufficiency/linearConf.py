import numpy as np
from sklearn.linear_model import LinearRegression
from lingam.resit import RESIT
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_data_with_latent_confounders(n_variables, n_samples, n_latent_confounders=2):
    """
    Generate data with both observed variables and latent confounders based on Chapter 2 of the paper.
    
    Args:
        n_variables: Number of observed variables.
        n_samples: Number of samples to generate.
        n_latent_confounders: Number of latent confounders influencing the observed variables.
        
    Returns:
        data: Generated data matrix (n_samples x n_variables).
        true_dag: Ground truth adjacency matrix for observed variables.
        confounder_effects: Dictionary of latent confounder effects.
    """
    # Create a DAG for the observed variables
    true_dag = np.zeros((n_variables, n_variables))
    for i in range(1, n_variables):
        parents = np.random.choice(range(i), size=np.random.randint(1, i + 1), replace=False)
        true_dag[i, parents] = 1

    # Generate coefficients for edges in the DAG
    coefficients = np.random.uniform(0.5, 2.0, size=true_dag.shape) * true_dag

    # Initialize data matrix
    data = np.zeros((n_samples, n_variables))

    # Generate data for observed variables
    for i in range(n_samples):
        for j in range(n_variables):
            parents = np.where(true_dag[j] == 1)[0]
            parent_data = np.sum(coefficients[j, parents] * data[i, parents]) if len(parents) > 0 else 0
            noise = np.random.uniform(-np.sqrt(3 * 0.5), np.sqrt(3 * 0.5))
            data[i, j] = parent_data + noise

    # Generate latent confounders and their effects
    latent_confounders = np.random.normal(0, 1, size=(n_samples, n_latent_confounders))
    confounder_effects = {}

    for confounder_idx in range(n_latent_confounders):
        affected_vars = np.random.choice(range(n_variables), size=2, replace=False)
        confounder_effect = np.random.uniform(0.5, 2.0, size=2)  # Effect size of confounder
        confounder_effects[confounder_idx] = {"vars": affected_vars, "effect": confounder_effect}
        
        # Add the latent confounder effect to the data
        for var, effect in zip(affected_vars, confounder_effect):
            data[:, var] += effect * (latent_confounders[:, confounder_idx] ** 2)  # Strong nonlinear effect

    return data, true_dag, confounder_effects

def compute_shd(true_dag, inferred_dag):
    """Compute Structural Hamming Distance (SHD) between two DAGs."""
    return np.sum(true_dag != inferred_dag)

def run_resit_analysis(data, true_dag, dag_type):
    """
    Run RESIT on the given data and compare with the ground truth.
    
    Args:
        data: Data matrix.
        true_dag: Ground truth adjacency matrix.
        dag_type: Type of DAG ("Base" or "Confounded").
        
    Returns:
        shd: Structural Hamming Distance (SHD) between inferred and true DAG.
        inferred_dag: Adjacency matrix inferred by RESIT.
    """
    logging.info(f"Applying RESIT to {dag_type} data...")
    model = RESIT(regressor=LinearRegression(), alpha=0.05)
    model.fit(data)

    inferred_dag = model.adjacency_matrix_
    shd = compute_shd(true_dag, inferred_dag)
    logging.info(f"SHD ({dag_type}): {shd}")
    logging.info(f"Inferred Adjacency Matrix ({dag_type}):\n{inferred_dag}")
    return shd, inferred_dag

def run_experiments(n_samples, n_variables, repetitions=100):
    """
    Run the experiment multiple times and summarize results.
    
    Args:
        n_samples: Number of samples to generate.
        n_variables: Number of observed variables.
        repetitions: Number of repetitions of the experiment.
        
    Returns:
        Summary of SHD results.
    """
    shd_confounded_list = []

    for i in range(repetitions):
        logging.info(f"--- Experiment {i+1}/{repetitions} ---")
        
        # Generate data with latent confounders
        data, true_dag, confounder_effects = generate_data_with_latent_confounders(n_variables, n_samples)

        # Apply RESIT to confounded data
        shd_confounded, inferred_confounded_dag = run_resit_analysis(data, true_dag, "Confounded")
        shd_confounded_list.append(shd_confounded)

    # Compute summary statistics
    avg_shd_confounded = np.mean(shd_confounded_list)
    std_shd_confounded = np.std(shd_confounded_list)

    print("\n--- Summary of Results ---")
    print(f"Variables: {n_variables}, SHD Confounded Mean ± Std: {avg_shd_confounded:.2f} ± {std_shd_confounded:.2f}")

    # Print all SHD values for later analysis
    print("\n--- Full SHD Results ---")
    print(f"Confounded SHD values: {shd_confounded_list}")

if __name__ == "__main__":
    run_experiments(n_samples=500, n_variables=4, repetitions=100)