import numpy as np

Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = np.array([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

W1 = np.array([
    [1.0, -1.0, 0.5],
    [0.5, 1.0, -0.5]
])

W2 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
])

# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Layer Normalization
def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    return (x - mean) / np.sqrt(variance + eps)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Attention Mechanism
scores = Q @ K.T
# Scale scores
scale_factor = np.sqrt(K.shape[-1])
# Normalize scores
norm_scores = scores / scale_factor
# Softmax to get attention weights
attention_weights = softmax(norm_scores)
# Attention output
attention_output = attention_weights @ V
# First residual connection
residual1 = X + attention_output
# Layer normalization
normalized1 = layer_norm(residual1)

hidden = relu(normalized1 @ W1)
ff_output = hidden @ W2

residual2 = normalized1 + ff_output
output = layer_norm(residual2)

# Output
print("Attention weights: \n", attention_weights)
print("Attention output: \n", attention_output)
print("After first residual: \n", residual1)
print("After first LayerNorm: \n", normalized1)
print("Feed-forward hidden: \n", hidden)
print("Feed-forward output: \n", ff_output)
print("After second residual: \n", residual2)
print("Final output: \n", output)