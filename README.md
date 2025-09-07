# Numerical Solutions of Non-linear Ordinary Differential Equations using Kolmogorov-Arnold Networks

## Overview

This repository contains the experiment for the manuscript "Numerical Solutions of Non-linear Ordinary Differential Equations using Kolmogorov-Arnold Networks".

The project implements a numerical framework that constructs KAN-based trial
solutions for ordinary differential equations and optimizes compact sets of
univariate basis functions to minimize the solution error. Benchmarks compare
KAN-based solvers to classical integrators (Euler, Runge–Kutta) and neural
approximators (RBFN, WNN).


## Repository structure

```
/README.md                                           # This file
/Codes/ Source Code/ Pipeline.ipynb                  # Reproducible pipeline and experiments (provided)
/ Codes / example_1_solution.png                     # Example 1 Solution
/ Codes / example_2_solution.png                     # Example 2 Solution
/ Codes / example_3_solution.png                     # Example 3 Solution
```

> Note: The repository contains `Pipeline.ipynb`. Open and run it to reproduce
> the main experiment flow.

## Requirements

* Python 3.9 or newer.
* Standard scientific stack: numpy, matplotlib, pandas, polars, scikit-learn, jax
* Deep learning framework: PyKAN.

Install with pip:

```bash
pip install numpy matplotlib torch pandas pykan jax scikit-learn polars
```



## Quick start

1. Activate the environment.
2. Open the pipeline notebook:

```bash
jupyter lab Pipeline.ipynb
# or
jupyter notebook Pipeline.ipynb
```

3. Run cells in order. The notebook orchestrates data generation, model
   construction, optimization, and plotting for the reported experiments.




## Implementation notes

* The method relies on the Kolmogorov–Arnold representation to express
  multivariate trial functions as a superposition of univariate basis
  functions.
* Optimization minimizes an ODE residual/error functional over the compact set
  of univariate basis parameters.
* Experiment modules include benchmark ODE definitions, numerical baselines,
  and KAN model wrappers.



## Citation

If you use this code cite the manuscript. Suggested citation format:

```
TBA
```

## License

[MIT License][https://github.com/IAmFarrokhnejad/Differential-Equation-Function-Approximation-Using-KAN/blob/main/LICENSE]

## Contact

Corresponding Author: Ali Farrokhnejad - ALI.FARROKHNEJAD@emu.edu.tr

---