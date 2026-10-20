import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

training_samples = 100

# range from -2 to 2
X = np.linspace(-2, 2, training_samples)

# true slope, intercept and noise
true_slope = 2.0
true_intercept = 1.0
true_noise = 0.3

noise = np.random.normal(0, true_noise, training_samples)
y = true_slope * X + true_intercept + noise

w = 0.0
b = 0.0
learning_rate = 0.01

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


def train_model(X, y, batch_size, epochs, learning_rate):
    w = 0.0
    b = 0.0

    loss_history = []

    for epoch in range(epochs):
        # Shuffle the training samples
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Split the shuffled data into batches
        batches = [X_shuffled[i:i+batch_size] for i in range(0, len(X_shuffled), batch_size)]
        batches_y = [y_shuffled[i:i+batch_size] for i in range(0, len(y_shuffled), batch_size)]

        for batch_X, batch_y in zip(batches, batches_y):
            # Calculate predictions
            y_pred = predict(batch_X, w, b)
            # Calculate gradients
            gradient_w, gradient_b = compute_gradients(batch_X, batch_y, y_pred)
            # Update w and b
            w = w - learning_rate * gradient_w
            b = b - learning_rate * gradient_b

        # Calculate MSE on the complete training dataset
        loss = compute_loss(y, predict(X, w, b))
        loss_history.append(loss)
    return w, b, loss_history


def train_with_accumulation(X, y, micro_batch_size, accumulation_steps, epochs, learning_rate):
    w = 0.0
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        # Shuffle X and y together
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        # Process micro-batches of 5 samples
        batches = [X_shuffled[i:i+micro_batch_size] for i in range(0, len(X_shuffled), micro_batch_size)]
        batches_y = [y_shuffled[i:i+micro_batch_size] for i in range(0, len(y_shuffled), micro_batch_size)]

        gradient_w_accumulated = 0.0
        gradient_b_accumulated = 0.0

        accumulated_samples = 0

        for batch_index, (batch_X, batch_y) in enumerate(zip(batches, batches_y), start=1):
            # Calculate predictions
            y_pred = predict(batch_X, w, b)
            # Calculate gradients
            gradient_w, gradient_b = compute_gradients(batch_X, batch_y, y_pred)
            # Accumulate sample-weighted gradients
            gradient_w_accumulated += gradient_w * len(batch_X)
            gradient_b_accumulated += gradient_b * len(batch_X)
            # Count accumulated samples
            accumulated_samples += len(batch_X)
            is_accumulation_complete = (batch_index % accumulation_steps == 0)
            is_last_batch = batch_index == len(batches)
            if is_accumulation_complete or is_last_batch:
                # Calculate average gradients
                gradient_w_accumulated /= accumulated_samples
                gradient_b_accumulated /= accumulated_samples
                # Update w and b once
                w = w - learning_rate * gradient_w_accumulated
                b = b - learning_rate * gradient_b_accumulated
                # Reset accumulated gradients
                gradient_w_accumulated = 0.0
                gradient_b_accumulated = 0.0
                accumulated_samples = 0
        # Calculate full-dataset MSE
        loss = compute_loss(y, predict(X, w, b))
        loss_history.append(loss)
    return w, b, loss_history

np.random.seed(123)
w_direct, b_direct, loss_direct = train_model(
    X, y, batch_size=20, epochs=100, learning_rate=0.01
)

np.random.seed(123)
w_accum, b_accum, loss_accum = train_with_accumulation(
    X, y,
    micro_batch_size=5,
    accumulation_steps=4,
    epochs=100,
    learning_rate=0.01
)

print("Direct batching:")
print(f"Weight: {w_direct:.6f}")
print(f"Bias: {b_direct:.6f}")
print(f"Final MSE: {loss_direct[-1]:.6f}")

print("\nGradient accumulation:")
print(f"Weight: {w_accum:.6f}")
print(f"Bias: {b_accum:.6f}")
print(f"Final MSE: {loss_accum[-1]:.6f}")

print("\nDifferences:")
print(f"Weight difference: {abs(w_direct - w_accum):.10f}")
print(f"Bias difference: {abs(b_direct - b_accum):.10f}")

batch_sizes = [1, 10, 50, 100]
results = {}
for batch_size in batch_sizes:
    # Reset the random seed for reproducibility
    np.random.seed(42)
    # Train the model with this batch size
    w, b, loss_history = train_model(X, y, batch_size=batch_size, epochs=100, learning_rate=0.01)
    # Store weights, bias and loss history
    # in the results dictionary
    results[batch_size] = {
        'weight': w,
        'bias': b,
        'loss_history': loss_history
    }

    # Print batch size, weight, bias and final MSE
    print(f"Batch size: {batch_size}")
    print(f"Weight: {w:.6f}")
    print(f"Bias: {b:.6f}")
    print(f"Final MSE: {loss_history[-1]:.6f}")


# Experiment 1: Compare different batch sizes
batch_sizes = [1, 10, 50, 100]
results = {}

for batch_size in batch_sizes:
    np.random.seed(42)

    w, b, loss_history = train_model(
        X, y,
        batch_size=batch_size,
        epochs=100,
        learning_rate=0.01
    )

    results[batch_size] = {
        "weight": w,
        "bias": b,
        "loss_history": loss_history
    }

# Experiment 2: Direct batching vs gradient accumulation
np.random.seed(123)
w_direct, b_direct, loss_direct = train_model(
    X, y,
    batch_size=20,
    epochs=100,
    learning_rate=0.01
)

np.random.seed(123)
w_accum, b_accum, loss_accum = train_with_accumulation(
    X, y,
    micro_batch_size=5,
    accumulation_steps=4,
    epochs=100,
    learning_rate=0.01
)

# Create two charts side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left chart: Compare batch sizes
ax1 = axes[0]

for batch_size in batch_sizes:
    loss_history = results[batch_size]["loss_history"]

    ax1.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        label=f"Batch Size = {batch_size}"
    )

ax1.set_title("Mini-Batch Training with Different Batch Sizes")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Mean Squared Error")
ax1.legend()
ax1.grid(True)

# Right chart: Direct batching vs accumulation
ax2 = axes[1]

ax2.plot(
    range(1, len(loss_direct) + 1),
    loss_direct,
    label="Direct Batch (20)",
    linewidth=2
)

ax2.plot(
    range(1, len(loss_accum) + 1),
    loss_accum,
    label="Gradient Accumulation (5 × 4)",
    linestyle="--",
    linewidth=2
)

ax2.set_title("Direct Batching vs Gradient Accumulation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Squared Error")
ax2.legend()
ax2.grid(True)

# Match Y-axis scales for easier visual comparison
ax1.set_ylim(bottom=0)
ax2.set_ylim(ax1.get_ylim())

# Add overall title
fig.suptitle(
    "Day 16: Mini-Batch Training and Gradient Accumulation",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

# Save the chart for GitHub and Medium
plt.savefig(
    "day_016_training_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
