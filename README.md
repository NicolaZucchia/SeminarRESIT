## Repository for conducting experiments with RESIT algorithm in various scenarios

The repository is organized as follows:

- regular_experiments : folder for the standard experiments
	- main.py : main script, files are passed by name
	- experiment_linear.py : data generation for linear case (Y = a * X + e)
	- experiment_quadratic.py : data generation for quadratic case (Y = a * X ** 2 + e)
	- experiment_peterslinear.py : simulation of linear setting described in Peters et al. (2014)

- violations_experiments : folder for the experiments where assumptions are violated to test the effect
	- violationsCausalSufficiency : folder for testing the effect of the presence of confounders
	- violationGaussianNoise : folder for testing whether using Gaussian noise instead of Uniform and the variance of noise impacts the linear setting (breaking Markov Equivalence Class)

- nonlinear_experiments: folder for running specifically nonlinear complex experiments
	the folder contains various version of debugging, the Pro one logs every step of the RESIT with also p-values and choices

- DAG_Maker.py : script to draw a DAG passing its adjacency matrix in input

# How to use the experiments

Inside the experiments the number of variables, samples and the repetitions can be manually adjusted to create various experiments. 
