import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class KANODESolver:
    def __init__(self, num_inner_neurons=10, num_outer_neurons=5, activation='sigmoid'):
        self.num_inner = num_inner_neurons
        self.num_outer = num_outer_neurons
        
        # Initialize weights and biases
        self.inner_weights = np.random.randn(num_inner_neurons)
        self.inner_biases = np.random.randn(num_inner_neurons)
        self.outer_weights = np.random.randn(num_outer_neurons, num_inner_neurons)
        self.outer_biases = np.random.randn(num_outer_neurons)
        self.final_weights = np.random.randn(num_outer_neurons)
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)
        else:
            self.activation = lambda x: np.tanh(x)
            self.activation_derivative = lambda x: 1 - x**2

    def forward(self, x):
        # Inner layer
        inner_outputs = self.activation(x * self.inner_weights + self.inner_biases)
        
        # Outer layer
        outer_outputs = []
        for i in range(self.num_outer):
            out = self.activation(np.sum(inner_outputs * self.outer_weights[i]) + self.outer_biases[i])
            outer_outputs.append(out)
            
        # Final aggregation
        return np.sum(np.array(outer_outputs) * self.final_weights)

    def get_derivative(self, x):
        # Compute derivative of the network output with respect to x
        h = 1e-7
        return (self.forward(x + h) - self.forward(x)) / h

    def get_parameters(self):
        # Flatten all parameters into a single array
        params = np.concatenate([
            self.inner_weights,
            self.inner_biases,
            self.outer_weights.flatten(),
            self.outer_biases,
            self.final_weights
        ])
        return params

    def set_parameters(self, params):
        # Update network parameters from flattened array
        idx = 0
        
        # Inner weights and biases
        self.inner_weights = params[idx:idx + self.num_inner]
        idx += self.num_inner
        
        self.inner_biases = params[idx:idx + self.num_inner]
        idx += self.num_inner
        
        # Outer weights
        outer_weights_size = self.num_outer * self.num_inner
        self.outer_weights = params[idx:idx + outer_weights_size].reshape(self.num_outer, self.num_inner)
        idx += outer_weights_size
        
        # Outer biases
        self.outer_biases = params[idx:idx + self.num_outer]
        idx += self.num_outer
        
        # Final weights
        self.final_weights = params[idx:idx + self.num_outer]

def solve_example1():
    # Example 1 from the paper
    def ode1(t, y):
        return 2*t + t**3 + t**2 * ((1 + 3*t**2)/(1 + t + t**3)) - y * (t + (1 + 3*t**2)/(1 + t + t**3))

    # Create KAN network
    kan = KANODESolver(num_inner_neurons=15, num_outer_neurons=7)
    
    # Define error function for optimization
    def error_func(params):
        kan.set_parameters(params)
        
        # Sample points for error calculation
        t_points = np.linspace(0, 1, 50)
        error = 0
        
        for t in t_points:
            y_pred = kan.forward(t)
            dy_pred = kan.get_derivative(t)
            error += 0.5 * (dy_pred - ode1(t, y_pred))**2
            
        # Add initial condition penalty
        ic_error = (kan.forward(0) - 1)**2
        return error + 1000 * ic_error

    # Optimize network parameters
    result = minimize(error_func, kan.get_parameters(), method='BFGS', options={'maxiter': 1000})
    kan.set_parameters(result.x)
    
    # Generate solution
    t_eval = np.linspace(0, 1, 21)
    y_pred = [kan.forward(t) for t in t_eval]
    
    return t_eval, y_pred

def solve_example2():
    # Example 2 from the paper
    def ode2(t, y):
        return np.cos(4*t) - 2*y

    # Create KAN network
    kan = KANODESolver(num_inner_neurons=20, num_outer_neurons=10)
    
    # Define error function for optimization
    def error_func(params):
        kan.set_parameters(params)
        
        # Sample points for error calculation
        t_points = np.linspace(0, 3, 100)
        error = 0
        
        for t in t_points:
            y_pred = kan.forward(t)
            dy_pred = kan.get_derivative(t)
            error += 0.5 * (dy_pred - ode2(t, y_pred))**2
            
        # Add initial condition penalty
        ic_error = (kan.forward(0) - 3)**2
        return error + 1000 * ic_error

    # Optimize network parameters
    result = minimize(error_func, kan.get_parameters(), method='BFGS', options={'maxiter': 1000})
    kan.set_parameters(result.x)
    
    # Generate solution
    t_eval = np.linspace(0, 3, 21)
    y_pred = [kan.forward(t) for t in t_eval]
    
    return t_eval, y_pred

# Solve and plot both examples
def main():
    # Example 1
    t1, y1 = solve_example1()
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t1, y1, 'b-', label='KAN Solution')
    plt.title('Example 1: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    
    # Example 2
    t2, y2 = solve_example2()
    
    plt.subplot(1, 2, 2)
    plt.plot(t2, y2, 'r-', label='KAN Solution')
    plt.title('Example 2: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()