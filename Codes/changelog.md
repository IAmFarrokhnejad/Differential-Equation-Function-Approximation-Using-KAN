# V 0.1:
Base implementation


# V 1.0:
Overhauled version


# V 1.1.0:
This implementation uses the **PyKAN** library directly and has several advantages over the previous version:

- Uses PyKAN's built-in optimization and training methods
- Takes advantage of PyKAN's specialized architecture for neural networks
- Uses automatic differentiation from PyTorch
- Includes grid-based interpolation with cubic splines

### Key changes from the previous version:
- Uses `MultKAN` class instead of custom implementation
- Leverages PyTorch's `autograd` for computing derivatives
- Uses `LBFGS` optimizer from PyKAN's built-in methods
- Includes grid-based interpolation parameters
- Uses PyKAN's custom training loop functionality


# V 1.1.1:
Bug fixes
### Key changes from the previous version:
- Removed the custom training loop and used PyKAN's standard fit method
- Created proper datasets with input-output pairs
- Added explicit handling of initial conditions after training
- Used regularization parameter (lamb) to control overfitting
- Simplified the ODE definitions to work with PyKAN's training method