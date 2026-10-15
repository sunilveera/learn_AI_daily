import numpy as np
import matplotlib.pyplot as plt

initial_learning_rate = 0.1
num_iterations = 100
decay_factor = 0.5
decay_interval = 20
momentum_beta = 0.9

weight_history = []
loss_history = []
learning_rate_history = []

# constant learning rate
def constant_lr(initial_lr):
    return initial_lr

# step decay learning rate
def step_decay_lr(initial_lr, iteration, decay_factor, decay_interval):
    if iteration > 1 and (iteration - 1) % decay_interval == 0:
            initial_lr = initial_lr * decay_factor
    return initial_lr

# cosine annealing learning rate
def cosine_annealing_lr(initial_lr, iteration, total_iterations):
    return (initial_lr/2) * (1 + np.cos(np.pi * iteration / total_iterations))

# function to calculate the gradient
def compute_gradient(w):
    return 2*(w-3)

# Defines a function to update the weight using the gradient descent algorithm
def update_weight(w, velocity):
    return w - velocity

def train_with_momentum(schedule_fn, num_iterations):
    weight = 0.0
    velocity = 0.0

    weight_history = []
    loss_history = []
    learning_rate_history = []

    learning_rate = initial_learning_rate
    for i in range(1, num_iterations+1):
        # Calculate current learning rate
        if schedule_fn == 'constant':
            learning_rate = constant_lr(initial_learning_rate)
        elif schedule_fn == 'step_decay':
            learning_rate = step_decay_lr(learning_rate, i, decay_factor, decay_interval)
        elif schedule_fn == 'cosine_annealing':
            learning_rate = cosine_annealing_lr(initial_learning_rate, i, num_iterations)
        else:
            raise ValueError(f"Invalid schedule function: {schedule_fn}")

        # Calculate the gradient
        gradient = compute_gradient(weight)
        # Update the velocity
        velocity = velocity * momentum_beta + learning_rate * gradient
        # Update the weight
        weight = update_weight(weight, velocity)
        # Calculate the loss
        loss = (weight-3)**2
        # Store weight, loss, learning rate
        weight_history.append(weight)
        loss_history.append(loss)
        learning_rate_history.append(learning_rate)
    return weight_history, loss_history, learning_rate_history

# train the model with the constant learning rate
constant_weight_history, constant_loss_history, constant_learning_rate_history = train_with_momentum('constant', num_iterations)
# train the model with the step decay learning rate
step_decay_weight_history, step_decay_loss_history, step_decay_learning_rate_history = train_with_momentum('step_decay', num_iterations)
# train the model with the cosine annealing learning rate
cosine_annealing_weight_history, cosine_annealing_loss_history, cosine_annealing_learning_rate_history = train_with_momentum('cosine_annealing', num_iterations)


# plot the loss histories
plt.axhline(y=0, color='gray', linestyle='--', label='Optimal Loss (0)')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(range(num_iterations), constant_loss_history, label='Constant Learning Rate', color='red')
plt.plot(range(num_iterations), step_decay_loss_history, label='Step Decay Learning Rate', color='blue')
plt.plot(range(num_iterations), cosine_annealing_loss_history, label='Cosine Annealing Learning Rate', color='green')
plt.legend()
plt.show()


# plot the weight histories
plt.axhline(y=3, color='gray', linestyle='--', label='Optimal Weight (3)')
plt.xlabel('Iteration')
plt.ylabel('Weight')
plt.plot(range(num_iterations), constant_weight_history, label='Constant Learning Rate', color='red')
plt.plot(range(num_iterations), step_decay_weight_history, label='Step Decay Learning Rate', color='blue')
plt.plot(range(num_iterations), cosine_annealing_weight_history, label='Cosine Annealing Learning Rate', color='green')
plt.legend()
plt.show()

# plot the learning rate histories
plt.axhline(y=0, color='gray', linestyle='--', label='Optimal Learning Rate (0)')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.plot(range(num_iterations), constant_learning_rate_history, label='Constant Learning Rate', color='red')
plt.plot(range(num_iterations), step_decay_learning_rate_history, label='Step Decay Learning Rate', color='blue')
plt.plot(range(num_iterations), cosine_annealing_learning_rate_history, label='Cosine Annealing Learning Rate', color='green')
plt.legend()
plt.show()
