#THE MAJOR ISSUE REMAINS AND THE CODE REQUIRES BUG FIXING 
import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import *

def solve_example1():
    """
    Solve the first ODE example:
    dy/dt + (t + (1 + 3t^2)/(1 + t + t^3))y = 2t + t^3 + t^2((1 + 3t^2)/(1 + t + t^3))
    y(0) = 1
    """
    # Create KAN model: 1D input (t), 1D output (y), 8 hidden neurons
    model = MultKAN(width=[1,8,1], grid=10, k=3, seed=42)
    
    # Define the ODE for training
    def ode1(t, y):
        return 2*t + t**3 + t**2 * ((1 + 3*t**2)/(1 + t + t**3)) - y * (t + (1 + 3*t**2)/(1 + t + t**3))
    
    # Create training points
    t_train = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # Create target values using the ODE
    with torch.no_grad():
        y_init = torch.ones_like(t_train)  # Initial guess
        target = ode1(t_train, y_init)
    
    # Create dataset
    dataset = {
        'train_input': t_train,
        'train_output': target,
        'train_label': target, #Added this as well 
        'valid_input': t_train,
        'valid_output': target, 
        'test_input': t_train, #placeholder # I ADDED THESE PLACEHOLDERS AND THE ALIAS ABOVE TO PREVENT 'KET ERRORS' 
        'test_output': target, #placeholder
    } #THE MAJOR ISSUE REMAINS AND THE CODE REQUIRES BUG FIXING 
    
    # Train the model
    model.fit(dataset, opt='LBFGS', steps=50, lamb=0.001)
    
    # Apply initial condition constraint
    with torch.no_grad():
        init_correction = 1.0 - model(torch.tensor([[0.0]]))
        model.nets[0].last.bias.data += init_correction
    
    # Generate solution points for plotting
    t_eval = torch.linspace(0, 1, 21).reshape(-1, 1)
    y_pred = model(t_eval).detach().numpy()
    
    return t_eval.numpy(), y_pred

def solve_example2():
    """
    Solve the second ODE example:
    dy/dt + 2y = cos(4t)
    y(0) = 3
    """
    # Create KAN model: 1D input (t), 1D output (y), 10 hidden neurons
    model = MultKAN(width=[1,10,1], grid=15, k=3, seed=42)
    
    # Define the ODE
    def ode2(t):
        return torch.cos(4*t)
    
    # Create training points
    t_train = torch.linspace(0, 3, 150).reshape(-1, 1)
    
    # Create target values using the ODE
    with torch.no_grad():
        target = ode2(t_train)
    
    # Create dataset
    dataset = {
        'train_input': t_train,
        'train_output': target,
        'valid_input': t_train,
        'valid_output': target
    }
    
    # Train the model
    model.fit(dataset, opt='LBFGS', steps=50, lamb=0.001)
    
    # Apply initial condition constraint
    with torch.no_grad():
        init_correction = 3.0 - model(torch.tensor([[0.0]]))
        model.nets[0].last.bias.data += init_correction
    
    # Generate solution points for plotting
    t_eval = torch.linspace(0, 3, 21).reshape(-1, 1)
    y_pred = model(t_eval).detach().numpy()
    
    return t_eval.numpy(), y_pred

def plot_results(t1, y1, t2, y2):
    plt.figure(figsize=(12, 5))
    
    # Plot Example 1
    plt.subplot(1, 2, 1)
    plt.plot(t1, y1, 'b-', label='KAN Solution')
    plt.title('Example 1: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    
    # Plot Example 2
    plt.subplot(1, 2, 2)
    plt.plot(t2, y2, 'r-', label='KAN Solution')
    plt.title('Example 2: Solution')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    print("Solving Example 1...")
    t1, y1 = solve_example1()
    
    print("Solving Example 2...")
    t2, y2 = solve_example2()
    
    print("Plotting results...")
    plot_results(t1, y1, t2, y2)

if __name__ == "__main__":
    main()

    #THE CODE REQUIRES BUG FIXING 