import numpy as np

np.random.seed(42)

vocab = ["cat", "dog", "runs", "sleeps"]

vocab_size = len(vocab)
d_model = 3
learning_rate = 0.1

# Fixed input representation for "the cat"
X = np.array([0.5, 0.2, 0.8])

# Correct next token
target_token = "sleeps"
target_id = vocab.index(target_token)

# Initialize a random trainable weight matrix W with shape (3, 4).
W = np.random.randn(d_model, vocab_size)

# Create a one-hot target vector of length vocab_size, containing 1 at target_id and 0 everywhere else.
target_vector = np.zeros(vocab_size)
target_vector[target_id] = 1

# softmax function
def softmax(x):
    # Subtract the maximum logit before exponentiation for numerical stability.
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Implement a function called cross_entropy_loss.
def cross_entropy_loss(softmax_logits, target_id):
    # Extract the predicted probability of the correct token.
    predicted_probability = softmax_logits[target_id]
    predicted_probability = np.clip(predicted_probability, 1e-12, 1.0)
    # Calculate the negative natural logarithm using NumPy.
    # avoiding log(0) in extreme numerical cases.
    loss = -np.log(predicted_probability)
    return loss

def forward_pass(X, W):
    # Compute the logits by multiplying X and W.

    # Assert the shapes of the matrices.
    assert W.shape == (d_model, vocab_size)

    logits = X @ W

    # Assert the shape of the logits.
    assert logits.shape == (vocab_size,)
    # Apply softmax to the logits.
    softmax_logits = softmax(logits)

    # Assert the sum of the softmax_logits is 1.0.
    assert np.isclose(np.sum(softmax_logits), 1.0)

    return softmax_logits

# Calculate the gradient of the weights.
def calculate_gradient_W(X, softmax_logits, target_vector):
    # Calculate gradient_logits by subtracting the target vector from softmax_logits.
    gradient_logits = softmax_logits - target_vector

    # Calculate gradient_W using np.outer().
    gradient_W = np.outer(X, gradient_logits)

    # Assert the shape of the gradient_W.
    assert gradient_W.shape == W.shape

    return gradient_W

# function to update 
# Run the forward pass using the current W.
# Calculate cross-entropy loss.
# Calculate gradient_W.
# Update W using gradient descent.
def calculate_and_update_W(X, W, target_id):
    # Run the forward pass using the current W.
    softmax_logits = forward_pass(X, W)
    # Calculate cross-entropy loss.
    loss = cross_entropy_loss(softmax_logits, target_id)
    # Calculate gradient_W.
    gradient_W = calculate_gradient_W(X, softmax_logits, target_vector)
    # Update W using gradient descent.
    W = W - learning_rate * gradient_W
    return W, loss, softmax_logits


# print the target token
print(f"Target token: {target_token}")

print(f"| Iteration  |    Loss    | Probability |")
for i in range(101):
    # Calculate and update W.
    W, loss, softmax_logits = calculate_and_update_W(X, W, target_id)
    # Print the iteration and the loss and probability for every 10 iterations.
    if i % 10 == 0:
        # print in a table format
        
        print(f"|------------|------------|-------------|")
        print(f"| {i:^10} | {loss:^10.4f} | {softmax_logits[target_id]:^10.4f}  |")

print(f"|------------|------------|-------------|")

