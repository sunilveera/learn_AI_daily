import numpy as np

np.random.seed(42)

vocab = ["cat", "dog", "runs", "sleeps"]

vocab_size = len(vocab)
d_model = 3
learning_rate = 0.1
hidden_dim = 4

# Fixed input representation for "the cat"
X = np.array([0.5, 0.2, 0.8])

# Correct next token
target_token = "sleeps"
target_id = vocab.index(target_token)

# softmax function
def softmax(x):
    # Subtract the maximum logit before exponentiation for numerical stability.
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Randomly initialize W1 (3,4)
W1 = np.random.randn(d_model, hidden_dim)

# Randomly initialize W2 (4,4)
W2 = np.random.randn(hidden_dim, vocab_size)

def forward_pass(X, W1, W2): # returns z1, hidden_state, logits
    # Compute the logits by multiplying X and W1
    z1 = X @ W1
    # Compute the hiddeen state with ReLU activation
    hidden_state = relu(z1)
    # Compute the logits by multiplying hidden state and W2
    logits = hidden_state @ W2
    # Compute the softmax of the logits
    softmax_logits = softmax(logits)

    return z1, hidden_state, logits, softmax_logits

# Implement a function called cross_entropy_loss.
def cross_entropy_loss(softmax_logits, target_id):
    # Extract the predicted probability of the correct token.
    predicted_probability = softmax_logits[target_id]
    predicted_probability = np.clip(predicted_probability, 1e-12, 1.0)
    # Calculate the negative natural logarithm using NumPy.
    loss = -np.log(predicted_probability)
    return loss

# Calculate the gradient of the output layer with respect to W2
def calculate_gradient_W2(hidden_state, softmax_logits, target_id):
    # Calculate the gradient of the loss with respect to the softmax logits
    gradient_softmax_logits = np.zeros_like(softmax_logits)
    gradient_softmax_logits[target_id] = 1
    gradient_softmax_logits = softmax_logits - gradient_softmax_logits
    # Calculate the gradient of the loss with respect to the hidden state
    gradient_W2 = np.outer(hidden_state, gradient_softmax_logits)
    return gradient_softmax_logits, gradient_W2

# Calculate the gradient of the hidden layer with respect to W1
def calculate_gradient_W1(X, gradient_z1, W1):
    # Calculate the gradient of the loss with respect to the z1
    gradient_W1 = np.outer(X, gradient_z1)
    return gradient_W1

def train_step(X, W1, W2, target_id):
    z1, hidden_state, logits, softmax_logits = forward_pass(X, W1, W2)
    loss = cross_entropy_loss(softmax_logits, target_id)
    gradient_logits, gradient_W2 = calculate_gradient_W2(hidden_state, softmax_logits, target_id)

    gradient_hidden_state = gradient_logits @ W2.T
    relu_derivative = (hidden_state > 0).astype(float)
    gradient_z1 = gradient_hidden_state * relu_derivative
    gradient_W1 = calculate_gradient_W1(X, gradient_z1, W1)

    return W1, W2, loss, gradient_W1, gradient_W2, softmax_logits


print("|------------|------------|-------------|")
print("| Epoch      | Loss       | Probability |")
print("|------------|------------|-------------|")
for i in range(100):
    W1, W2, loss, gradient_W1, gradient_W2, softmax_logits = train_step(X, W1, W2, target_id)
    W1 = W1 - learning_rate * gradient_W1
    W2 = W2 - learning_rate * gradient_W2

    # print the new loss for each 10 iterations and the predicted probability of the correct token in table format
    if (i+1) % 10 == 0:
        print(f"| {i+1:^10} | {loss:^10.4f} | {softmax_logits[target_id]:^11.4f} |")
        
print("|------------|------------|-------------|")

