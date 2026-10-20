import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
training_samples = 100
# range from -10 to 10
X = np.linspace(-10, 10, training_samples)

# true slope, intercept and noise
true_slope = 2.0
true_intercept = 1.0
true_noise = 0.3

w = 0.0
b = 0.0
learning_rate = 0.01
num_epochs = 50

# Generate y values using a linear function with Gaussian noise
noise = np.random.normal(0, true_noise, training_samples)
y = true_slope * X + true_intercept + noise


def predict(X, w, b):
    return w * X + b

def compute_loss(y_true, y_pred):
    # Mean Squared Error
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    # Calculate prediction errors
    errors = y_pred - y_true
    # Calculate gradient_w
    gradient_w = 2 * np.mean(errors * X)
    # Calculate gradient_b
    gradient_b = 2 * np.mean(errors)
    # Return both gradients
    return gradient_w, gradient_b

# Training without gradient clipping.
loss_history = []
gradient_norm_history = []
weight_history = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = predict(X, w, b)
    # Compute loss
    loss = compute_loss(y, y_pred)
    # Compute gradients
    gradient_w, gradient_b = compute_gradients(X, y, y_pred)
    # Compute gradient norm
    gradient_norm = np.sqrt(gradient_w**2 + gradient_b**2)
    # Append to history
    loss_history.append(loss)
    gradient_norm_history.append(gradient_norm)
    weight_history.append((w, b))
    # Update weights and bias
    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

# Print the epoch number, loss, gradient norm, weight, and bias for the first 10 epochs
for i in range(10):
    print(f"Epoch {i+1}: Loss: {loss_history[i]}, Gradient Norm: {gradient_norm_history[i]}, Weight: {weight_history[i][0]}, Bias: {weight_history[i][1]}")


# Training with gradient clipping by value.
w_value = 0.0 
b_value = 0.0
loss_history_clip = []
gradient_norm_history_clip = []
weight_history_clip = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = predict(X, w_value, b_value)
    # Compute loss
    loss = compute_loss(y, y_pred)
    # Compute gradients
    gradient_w, gradient_b = compute_gradients(X, y, y_pred)
    # Clip gradients to a value of -5 to 5
    clipped_gradient_w = np.clip(gradient_w, -5, 5)
    clipped_gradient_b = np.clip(gradient_b, -5, 5)
    # Compute gradient norm
    gradient_norm = np.sqrt(clipped_gradient_w**2 + clipped_gradient_b**2)
    # Append to history
    loss_history_clip.append(loss)
    gradient_norm_history_clip.append(gradient_norm)
    weight_history_clip.append((w_value, b_value))
    # Update weights and bias
    w_value -= learning_rate * clipped_gradient_w
    b_value -= learning_rate * clipped_gradient_b

# Print the epoch number, loss, gradient norm, weight, and bias for the first 10 epochs
for i in range(10):
    print(f"Epoch {i+1}: Loss: {loss_history_clip[i]}, Gradient Norm: {gradient_norm_history_clip[i]}, Weight: {weight_history_clip[i][0]}, Bias: {weight_history_clip[i][1]}")


# Training with gradient clipping by global norm.
w_norm = 0.0
b_norm = 0.0
loss_history_norm = []
gradient_norm_history_norm = []
weight_history_norm = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = predict(X, w_norm, b_norm)
    # Compute loss
    loss = compute_loss(y, y_pred)
    # Compute gradients
    gradient_w, gradient_b = compute_gradients(X, y, y_pred)
    # Calculate their combined norm
    gradient_norm = np.sqrt(gradient_w**2 + gradient_b**2)
    # Apply a common scaling factor only when the norm exceeds max_norm = 5
    if gradient_norm > 5:
        scaling_factor = 5 / gradient_norm
        gradient_w *= scaling_factor
        gradient_b *= scaling_factor
    # Compute gradient norm
    clipped_gradient_norm = np.sqrt(gradient_w**2 + gradient_b**2)
    # Append to history
    loss_history_norm.append(loss)
    gradient_norm_history_norm.append(clipped_gradient_norm)
    weight_history_norm.append((w_norm, b_norm))
    # Update weights and bias
    w_norm -= learning_rate * gradient_w
    b_norm -= learning_rate * gradient_b

# Print the epoch number, loss, gradient norm, weight, and bias for the first 10 epochs 
for i in range(10):
    print(f"Epoch {i+1}: Loss: {loss_history_norm[i]}, Gradient Norm: {gradient_norm_history_norm[i]}, Weight: {weight_history_norm[i][0]}, Bias: {weight_history_norm[i][1]}")



# ============================================
# Day 17 - Gradient Clipping Comparison
# ============================================

epochs = np.arange(1, num_epochs + 1)

# Extract weight values from (weight, bias) histories
weights_no_clip = [item[0] for item in weight_history]
weights_value = [item[0] for item in weight_history_clip]
weights_norm = [item[0] for item in weight_history_norm]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --------------------------------------------
# Chart 1: Training Loss
# --------------------------------------------
axes[0].plot(epochs, loss_history,
             label="No Clipping", color="red")

axes[0].plot(epochs, loss_history_clip,
             label="Value Clipping", color="blue")

axes[0].plot(epochs, loss_history_norm,
             label="Global Norm Clipping", color="green")

axes[0].set_yscale("log")
axes[0].set_title("Training Loss Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss (Log Scale)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# --------------------------------------------
# Chart 2: Gradient Norm Comparison
# --------------------------------------------
axes[1].plot(epochs, gradient_norm_history,
             label="No Clipping", color="red")

axes[1].plot(epochs, gradient_norm_history_clip,
             label="Value Clipping", color="blue")

axes[1].plot(epochs, gradient_norm_history_norm,
             label="Global Norm Clipping", color="green")

axes[1].axhline(y=5, color="black", linestyle="--",
                label="Threshold = 5")

axes[1].set_yscale("log")
axes[1].set_title("Gradient Norm Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Gradient Norm (Log Scale)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


# --------------------------------------------
# Chart 3: Weight Convergence
# --------------------------------------------
# Exclude no-clipping because its weights diverge
# to extremely large values.

axes[2].plot(epochs, weights_value,
             label="Value Clipping", color="blue")

axes[2].plot(epochs, weights_norm,
             label="Global Norm Clipping", color="green")

axes[2].axhline(y=true_slope, color="red",
                linestyle="--", label="True Weight = 2")

axes[2].set_title("Weight Convergence Comparison")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Weight Value")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# --------------------------------------------
# Final layout and save
# --------------------------------------------
fig.suptitle(
    f"Day 17 - Gradient Clipping (Learning Rate = {learning_rate})",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    f"day_017_gradient_clipping_lr_{learning_rate}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
