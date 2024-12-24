
## V 1.1: 
#### Optimizations Applied:
- Added a loss term (ode_residual) that enforces the ODE dynamics.
- Used the LBFGS optimizer, which is effective for tasks requiring accurate gradient-based optimization.
- Limited training to 150 epochs with early stopping to prevent overfitting.
- Compared the predicted solution (y_pred) with the exact solution (y_exact_eval) for both examples.
#### V 1.1 Results:
~~Example one remians inaccurate, but it solves example two in a perfect manner.~~
Both examples are solved perfectly. They can be optimized further.


## V 1.3:
- Added example 3 from the second reference
#### V 1.3 Results:
Each example works fine, however, there's always room for improvement.

## V 1.3.1:
- Added dynamic learning rate increase and increased epochs, standard modifications for possible improvement.
#### V 1.3.1 Results:
- No noticeable improvements

## V 2.0:
- Fixed loss function, now loss is calculated as it should be.
## V 2.1:
- Fixed an issue with example 3's exact solution.
#### V 2.0-2.1 Results:
- Model possesses comparable loss to that of the references.
## V 2.2:
- Added comparison table for comparing network output and exact solutions

## **Next step:**
The network eventually halts and almost none of the optimization strategies have any impact on the accuracy. References below might prove useful for its optimization:
- https://github.com/lif314/X-KANeRF\
- https://file.notion.so/f/f/76051f0b-02ea-4789-9f79-63eaf64f43bb/ac6432e8-8269-4281-b2e5-f23f588fae0a/2410.03801v2.pdf?table=block&id=1622e363-46a2-8051-9800-d559954b0cb9&spaceId=76051f0b-02ea-4789-9f79-63eaf64f43bb&expirationTimestamp=1734890400000&signature=65SRNRMgzT3T7Yz62t3H4lCBf-H8f8cdNRo10Ib7wZE&downloadName=2410.03801v2.pdf

### Optimization Strategies(TODO):

#### 1. **Refine the Model Architecture**
- **Increase the Number of Neurons**: Use more neurons in the hidden layers to enhance the model’s capacity to capture complex relationships in the solution.
- **Adjust the Grid and Polynomial Order (`k`)**: For KANs, experiment with more grid points or higher-order polynomials to better approximate the underlying function.
- **Add More Layers**: Adding layers may help the network learn deeper relationships. However, this must be balanced to avoid overfitting.


---

#### 2. **Improved Loss Functions**
- **Weight the Loss Terms**:
  - Use a weighted combination of the physics-informed loss (ODE residual) and the boundary loss:
    \[
    \text{Total Loss} = \alpha \cdot \text{Physics Loss} + \beta \cdot \text{Boundary Loss},
    \]
    where \(\alpha, \beta > 0\) can be tuned.
  - Start with a higher weight for the boundary condition and reduce it as training progresses.
- **Additional Regularization**:
  - Add L2 regularization on the network weights to prevent overfitting.
  - Include a penalty term for the gradient to enforce smoothness:
    \[
    \text{Smoothness Loss} = \lambda \cdot \text{MSE}\left(\frac{\partial^2 y(t)}{\partial t^2}\right),
    \]
    where \(\lambda > 0\) is a hyperparameter.

---

#### 3. **Sampling Strategies**
- **Adaptive Sampling**:
  - Focus on regions where the residual of the ODE is large by dynamically adjusting the sampling points during training.
  - For instance, start with a uniform grid and later sample more points near sharp changes or high residuals.
- **Boundary Emphasis**:
  - Sample additional points near \(t = 0\) to ensure the network learns the boundary condition effectively.

---

#### 4. **Optimization and Training Techniques**
- **Switch to a Second-Order Optimizer**:
  - Use optimizers like L-BFGS for better convergence in physics-informed tasks, as it captures curvature information.
- **Learning Rate Scheduling**:
  - Start with a higher learning rate and gradually decrease it using a cosine annealing or exponential decay schedule.
- **Early Stopping**:
  - Monitor the validation loss and stop training when improvement plateaus, avoiding overfitting.
- **Gradient Clipping**:
  - Clip gradients to prevent instability caused by large updates.

---

#### 5. **Normalization and Scaling**
- **Input Normalization**:
  - Normalize \(t\) to \([-1, 1]\) instead of \([0, 1]\) for better numerical stability.
- **Output Scaling**:
  - Scale the expected solution (e.g., dividing by the maximum value) to make it easier for the model to learn.

---

#### 6. **Hyperparameter Tuning**
- Experiment with different hyperparameters for:
  - Number of neurons and layers.
  - KAN grid size and polynomial degree.
  - Learning rate and batch size.

---

#### 7. **Better Initialization**
- Initialize weights and biases carefully, using schemes like Xavier or He initialization, to prevent poor starting points.
- Pre-train the network using simpler functions (e.g., approximate solutions) before fine-tuning with the ODE loss.

---

#### 8. **Model Interpretability**
- Monitor the residuals of the ODE over the training domain to identify problematic regions.
- Plot intermediate outputs to ensure the model captures the general shape of the solution.

---

#### 9. **Incorporate Prior Knowledge**
- Use known properties of the solution (e.g., its monotonicity or asymptotic behavior) to guide the training process.
- Define custom activation functions tailored to the problem, such as radial basis functions or sigmoidal activations with specific properties.

---

#### 10. **Increase Training Iterations**
- Extend training iterations with periodic re-initialization of the optimizer to avoid local minima.

---