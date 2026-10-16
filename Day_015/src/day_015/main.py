
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# training inputs: 15 evenly spaced points from 0 to 2*pi
x_train = np.linspace(0, 2*np.pi, 15)

# random noise to the true sine values
y_train = np.sin(x_train) + np.random.normal(0, 0.2, 15)

# validation inputs: 100 evenly spaced points
x_val = np.linspace(0, 2*np.pi, 100)

# noisy validation targets independently
y_val = np.sin(x_val) + np.random.normal(0, 0.2, 100)

# Normalize both datasets using the same transformation
x_train_scaled = (x_train - np.pi) / np.pi
x_val_scaled = (x_val - np.pi) / np.pi


def polynomial_features(x, degree):
    # Generate columns for x^0, x^1, ..., x^degree
    return np.vander(x, degree+1, increasing=True)


def train_model(x, y, degree):
    # Build polynomial features
    X = polynomial_features(x, degree)
    weights = np.linalg.lstsq(X, y, rcond=None)[0]
    return weights


def predict(x, weights):
    # Determine degree from weights
    degree = weights.shape[0] - 1
    X = polynomial_features(x, degree)
    return X @ weights


def mse(y_true, y_pred):
    # Calculate mean squared error
    return np.mean((y_true - y_pred) ** 2)


# L2 regularization
def train_model_l2(x, y, degree, lambda_value):
    X = polynomial_features(x, degree)
    n = len(x)
    # Create identity matrix
    I = np.eye(degree + 1)
    # Do not regularize the bias
    I[0, 0] = 0
    # Calculate X.T @ X
    XX = X.T @ X
    # Add n * lambda_value * I
    XX += n * lambda_value * I
    # Solve the system of equations
    weights = np.linalg.solve(XX, X.T @ y)
    return weights


# calculate training and validation losses for the three models
training_losses = []
validation_losses = []
for degree in [1, 3, 12]:
    weights = train_model(x_train, y_train, degree)
    training_losses.append(mse(y_train, predict(x_train, weights)))
    validation_losses.append(mse(y_val, predict(x_val, weights)))
    print(f"Degree {degree} training loss: {training_losses[-1]}, validation loss: {validation_losses[-1]}")

# compare Degree 12 models using the following L2 strengths
lambda_values = [0.0, 0.0001, 0.001, 0.01, 0.1]


l2_training_losses = []
l2_validation_losses = []
for lambda_value in lambda_values:
    if lambda_value == 0.0:
        weights = train_model(x_train_scaled, y_train, 12)
    else:
        weights = train_model_l2(x_train_scaled, y_train, 12, lambda_value)
    l2_training_losses.append(mse(y_train, predict(x_train_scaled, weights)))
    l2_validation_losses.append(mse(y_val, predict(x_val_scaled, weights)))
    print(f"L2 regularization with lambda = {lambda_value} training loss: {l2_training_losses[-1]}, validation loss: {l2_validation_losses[-1]}")


# Compare all five L2 regularization strengths
lambda_values = [0.0, 0.0001, 0.001, 0.01, 0.1]

x_plot = np.linspace(0, 2 * np.pi, 300)
x_plot_scaled = (x_plot - np.pi) / np.pi

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for i, lambda_value in enumerate(lambda_values):
    ax = axes[i]

    # Train model
    if lambda_value == 0.0:
        weights = train_model(x_train_scaled, y_train, 12)
    else:
        weights = train_model_l2(
            x_train_scaled, y_train, 12, lambda_value
        )

    # Predictions
    y_pred = predict(x_plot_scaled, weights)
    train_loss = mse(
        y_train, predict(x_train_scaled, weights)
    )
    val_loss = mse(
        y_val, predict(x_val_scaled, weights)
    )

    # Plot training data
    ax.scatter(
        x_train, y_train,
        color="black", s=30, label="Training Data"
    )

    # True sine function
    ax.plot(
        x_plot, np.sin(x_plot),
        "--", color="blue", label="True Sine"
    )

    # Highlight lambda = 0.001
    is_best = lambda_value == 0.001

    ax.plot(
        x_plot, y_pred,
        color="green" if is_best else "orange",
        linewidth=3 if is_best else 1.8,
        label="Model Prediction"
    )

    ax.set_title(
        f"Lambda = {lambda_value}\n"
        f"Train MSE = {train_loss:.4f} | "
        f"Val MSE = {val_loss:.4f}",
        fontsize=12,
        fontweight="bold" if is_best else "normal",
        color="green" if is_best else "black"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_ylim(-2, 2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    if is_best:
        ax.set_facecolor("#e8f5e9")

        for spine in ax.spines.values():
            spine.set_color("green")
            spine.set_linewidth(3)

        ax.text(
            0.5, 0.04,
            "BEST VALIDATION LOSS",
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="green",
            bbox=dict(
                facecolor="white",
                edgecolor="green",
                boxstyle="round,pad=0.4"
            )
        )

# Hide unused sixth subplot
axes[5].axis("off")

fig.suptitle(
    "Day 15 — Effect of L2 Regularization on Overfitting",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("l2_regularization_comparison.png", dpi=300)
plt.show()

