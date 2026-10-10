import numpy as np

# using a seed for reproducibility
np.random.seed(42)

# vocabulary of the tokens
vocab = ["the", "cat", "sleeps", "runs", "dog", "eats", "a", "on", "mat", "."]

# size of the vocabulary
vocab_size = len(vocab)

# tokens of the sequence
tokens = ["the", "cat"]
token_ids = [vocab.index(token) for token in tokens]

sequence_length = len(token_ids)
d_model = 8
num_heads = 2
d_head = d_model // num_heads
d_ff = 16

# function for scaled dot-product attention
def attention(Q, K, V, mask=None):
    # 1. Calculate scaled attention scores
    scores = Q @ K.T / np.sqrt(d_head)
    # 2. Apply optional causal mask
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    # 3. Calculate numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    # 4. Multiply attention weights by V
    output = weights @ V
    # 5. Return attention weights and output
    return weights, output

# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Layer Normalization
def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # Return the normalized output
    return (x - mean) / np.sqrt(variance + eps)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Creates a randomly initialized token embedding matrix with shape (10, 8).
W_e = np.random.randn(vocab_size, d_model)

# Asserting the shape of the token embedding matrix.
assert W_e.shape == (vocab_size, d_model)

# Selects the embeddings for the tokens "the" and "cat", producing shape (2, 8).
X = W_e[token_ids]

# Creates positional embeddings for the two positions, also with shape (2, 8). 
# Using a simple learned-style randomly initialized positional embedding table.
W_p = np.random.randn(sequence_length, d_model)

# Asserting the shape of the positional embedding matrix.
assert W_p.shape == (sequence_length, d_model)

# Adds token and positional embeddings to create X.
X = X + W_p

# Prints the shapes and adds assertions verifying them.
print("X: \n", X)
print("X.shape: \n", X.shape)
assert X.shape == (sequence_length, d_model)

# Create separate random W_q, W_k, and W_v projection matrices for each head, each with shape (8, 4).
# projection matrices for head 1
W_q_1 = np.random.randn(d_model, d_head)
W_k_1 = np.random.randn(d_model, d_head)
W_v_1 = np.random.randn(d_model, d_head)

# projection matrices for head 2
W_q_2 = np.random.randn(d_model, d_head)
W_k_2 = np.random.randn(d_model, d_head)
W_v_2 = np.random.randn(d_model, d_head)

# Compute Q, K, and V using X.
Q_1 = X @ W_q_1
K_1 = X @ W_k_1
V_1 = X @ W_v_1
Q_2 = X @ W_q_2
K_2 = X @ W_k_2
V_2 = X @ W_v_2

# Create one causal mask using np.triu() and reuse it for both heads.
mask = np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)

# Apply the scaled dot-product attention function to each head.
weights_1, output_1 = attention(Q_1, K_1, V_1, mask)
weights_2, output_2 = attention(Q_2, K_2, V_2, mask)

# Verify each attention row sums to 1
assert np.allclose(weights_1.sum(axis=-1), 1.0)
assert np.allclose(weights_2.sum(axis=-1), 1.0)

# Verify future tokens receive zero attention
assert np.all(weights_1[mask] == 0)
assert np.all(weights_2[mask] == 0)

# Print attention matrices
print("Head 1 attention weights:\n", weights_1)
print("Head 2 attention weights:\n", weights_2)

# Concatenate the head outputs using np.concatenate(..., axis=-1).
output = np.concatenate([output_1, output_2], axis=-1)

# Asserting the shape of the concatenated output.
assert output.shape == (sequence_length, d_model)

# Create an output projection matrix W_o with shape (8, 8) and apply it to the concatenated output.
W_o = np.random.randn(d_model, d_model)
output = output @ W_o

# Asserting the shape of the output.
assert output.shape == (sequence_length, d_model)

# Print the output.
print("Output: \n", output)

# First residual connection 
first_residual_output = output + X
# and apply LayerNorm
first_residual_output = layer_norm(first_residual_output)

# Print the output.
print("Output after first residual connection and LayerNorm: \n", first_residual_output)

# Feed-Forward Network
# Create a randomly initialized projection matrix for the hidden layer with shape (8, 16) and (16, 8).
W_1 = np.random.randn(d_model, d_ff)
W_2 = np.random.randn(d_ff, d_model)

# Conceptual structure only
feed_forward_output = relu(first_residual_output @ W_1) @ W_2

# Second residual connection and LayerNorm
second_residual_output = feed_forward_output + first_residual_output
second_residual_output = layer_norm(second_residual_output)

# Print the output.
print("Output after second residual connection and LayerNorm: \n", second_residual_output)

assert first_residual_output.shape == (sequence_length, d_model)
assert feed_forward_output.shape == (sequence_length, d_model)
assert second_residual_output.shape == (sequence_length, d_model)

assert np.all(np.isfinite(second_residual_output))

assert np.allclose(
    second_residual_output.mean(axis=-1), 0.0, atol=1e-6
)

# Create a random vocabulary projection matrix W_vocab with shape (8, 10).
W_vocab = np.random.randn(d_model, vocab_size)

# Multiply second_residual_output by W_vocab to obtain vocabulary logits with shape (2, 10).
logits = second_residual_output @ W_vocab

# Select the last row of logits, representing the prediction after "the cat".
last_row_logits = logits[-1, :]

# Print the last row of logits.
print("Last row of logits: \n", last_row_logits)

# Apply softmax to that vector., 
# Improving its numerical stability by subtracting the maximum logit before exponentiation.
softmax_logits = softmax(last_row_logits - np.max(last_row_logits))

# Asserting the shapes of the matrices.
assert W_vocab.shape == (d_model, vocab_size)
assert logits.shape == (sequence_length, vocab_size)
assert last_row_logits.shape == (vocab_size,)
assert softmax_logits.shape == (vocab_size,)

assert np.all(np.isfinite(softmax_logits))
assert np.allclose(softmax_logits.sum(), 1.0)
assert np.all(softmax_logits >= 0)


# Print the softmax logits.
print("Softmax logits: \n", softmax_logits)

# Print the softmax logits and the corresponding token strings in a table format.
print("Softmax logits and corresponding token strings:")
print("--------------------------------")
print("| Token ID | Token String | Softmax Logit |")
print("--------------------------------")
# print the table with spaces between the columns.
for i in range(vocab_size):
    print(f"| {i:<8} | {vocab[i]:<12} | {softmax_logits[i]*100:<6.2f}% |")
print("--------------------------------")

# Use np.argmax() to select the highest-probability token ID.
predicted_token_id = np.argmax(softmax_logits)

# Asserting the predicted token ID is the highest-probability token ID.
assert predicted_token_id == np.argmax(last_row_logits)

# Convert the predicted token ID back into its token string using vocab.
predicted_token = vocab[predicted_token_id]

# Print the predicted token ID and token string.
print(f"Predicted token ID is [{predicted_token_id}] and token string is '{predicted_token}'")
