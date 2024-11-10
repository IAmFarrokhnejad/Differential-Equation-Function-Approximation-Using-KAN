# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define the Kolmogorov-Arnold Network model
# class KolmogorovArnoldNet(nn.Module):
#     def __init__(self, num_hidden_units):
#         super(KolmogorovArnoldNet, self).__init__()
#         self.hidden = nn.Linear(1, num_hidden_units)
#         self.output = nn.Linear(num_hidden_units, 1)
    
#     def forward(self, x):
#         h = torch.sin(self.hidden(x))  # Nonlinear transformation
#         return self.output(h)

# # Define the trial solution for the ODEs based on KAN
# def trial_solution(x, net, y0, example_num):
#     if example_num == 1:
#         return y0 + (x * net(x))
#     elif example_num == 2:
#         return y0 + (x ** 2 * net(x))

# # Define the loss function based on the differential equation of each example
# def differential_loss(x, net, example_num):
#     x = x.requires_grad_(True)
#     y_pred = trial_solution(x, net, y0=1 if example_num == 1 else 3, example_num=example_num)
#     y_pred_prime = torch.autograd.grad(y_pred, x, torch.ones_like(x), create_graph=True)[0]
    
#     if example_num == 1:
#         f = (2 * x + x**3 + x**2 * (1 + 3 * x**2 / (1 + x + x**3)))  # Example 1 target function
#         g = (x + 1 + 3 * x**2 / (1 + x + x**3)) * y_pred
#     elif example_num == 2:
#         f = torch.cos(4 * x)  # Example 2 target function
#         g = 2 * y_pred
        
#     loss = ((y_pred_prime + g - f) ** 2).mean()
#     return loss

# # Training function
# def train_model(net, x_train, example_num, epochs=1000, lr=0.01):
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         loss = differential_loss(x_train, net, example_num)
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {loss.item()}')
    
#     return net

# # Example 1: ODE solution
# x_train = torch.linspace(0, 1, 100).view(-1, 1)
# net_example1 = KolmogorovArnoldNet(num_hidden_units=10)
# trained_net_example1 = train_model(net_example1, x_train, example_num=1)

# # Example 2: ODE solution
# x_train = torch.linspace(0, 3, 100).view(-1, 1)
# net_example2 = KolmogorovArnoldNet(num_hidden_units=15)
# trained_net_example2 = train_model(net_example2, x_train, example_num=2)

# # Predict and evaluate
# x_test = torch.linspace(0, 1, 100).view(-1, 1)  # For example 1
# y_pred_example1 = trial_solution(x_test, trained_net_example1, y0=1, example_num=1).detach().numpy()

# x_test = torch.linspace(0, 3, 100).view(-1, 1)  # For example 2
# y_pred_example2 = trial_solution(x_test, trained_net_example2, y0=3, example_num=2).detach().numpy()

# # Print or plot predictions for comparison
# print("Example 1 Predictions:", y_pred_example1)
# print("Example 2 Predictions:", y_pred_example2)

from kan import *
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = MultKAN(width=[2,5,1], grid=5, k=3, seed=0)

# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)

# plot KAN at initialization
model(dataset['train_input'])
model.plot(beta=100)

# train the model
model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001)

model.plot()
model.prune()
