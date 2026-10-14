import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.1
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

num_iterations = 100
sgd_loss_history = []
momentum_loss_history = []
adam_loss_history = []
adam_m_history = []
adam_v_history = []
# function to calculate 
def compute_loss(w):
    return (w-3)**2

# function to calculate the gradient
def compute_gradient(w):
    return 2*(w-3)

# Defines a function to update the weight using the gradient descent algorithm
def update_weight(w, learning_rate):
    return w - learning_rate * compute_gradient(w)

# function to update the weight using the momentum algorithm
def update_weight_momentum(w, velocity, learning_rate, beta):
    gradient = compute_gradient(w)
    # velocity using momentum algorithm
    velocity = beta * velocity + learning_rate * gradient
    # update the weight
    w = w - velocity
    # return the updated weight and velocity
    return w, velocity

# function to update the weight using the adam algorithm
def update_weight_adam(w, m, v, learning_rate, beta1, beta2, epsilon, i):
    # calculate the gradient
    gradient = compute_gradient(w)
    # m using adam algorithm
    m = beta1 * m + (1 - beta1) * gradient
    # v using adam algorithm
    v = beta2 * v + (1 - beta2) * gradient**2
    # bias correction
    m_hat = m / (1 - beta1**(i+1))
    v_hat = v / (1 - beta2**(i+1))
    # update the weight
    w = w - learning_rate * m_hat / (epsilon + np.sqrt(v_hat))
    # return the updated weight and velocity
    return w, m, v

# SGD
weight = 0.0
for i in range(num_iterations):
    weight = update_weight(weight, learning_rate)
    # print(f"Iteration {i+1}: Weight = {weight}, Loss = {compute_loss(weight)}")
    sgd_loss_history.append(compute_loss(weight))

# Momentum
weight = 0.0
velocity = 0.0
for i in range(num_iterations):
    weight, velocity = update_weight_momentum(weight, velocity, learning_rate, beta)
    # print(f"Iteration {i+1}: Weight = {weight}, Loss = {compute_loss(weight)}")
    momentum_loss_history.append(compute_loss(weight))

# Adam
weight = 0.0
m = 0.0
v = 0.0
for i in range(num_iterations):
    weight, m, v = update_weight_adam(weight, m, v, learning_rate, beta1, beta2, epsilon, i)
    # print(f"Iteration {i+1}: Weight = {weight}, Loss = {compute_loss(weight)}")
    adam_loss_history.append(compute_loss(weight))
    adam_m_history.append(m)
    adam_v_history.append(v)

plt.plot(sgd_loss_history, label='SGD')
plt.plot(momentum_loss_history, label='Momentum')
plt.plot(adam_loss_history, label='Adam')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.show()