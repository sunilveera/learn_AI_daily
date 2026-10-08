import numpy as np

# number of tokens in the sequence
sequence_length = 3
# dimension of the model
d_model = 8
# number of heads in the multi-head attention
num_heads = 2
# dimension of each head
d_head = d_model // num_heads

# using a seed for reproducibility
np.random.seed(42)

# create a matrix of token embeddings
X = np.random.randn(sequence_length, d_model)
print("X: \n", X)

# projection matrices for head 1
W_q_1 = np.random.randn(d_model, d_head)
W_k_1 = np.random.randn(d_model, d_head)
W_v_1 = np.random.randn(d_model, d_head)

# projection matrices for head 2
W_q_2 = np.random.randn(d_model, d_head)
W_k_2 = np.random.randn(d_model, d_head)
W_v_2 = np.random.randn(d_model, d_head)

# projection matrix for the output
W_o = np.random.randn(d_model, d_model)

# project the token embeddings into the query, key, and value vectors
Q_1 = X @ W_q_1
K_1 = X @ W_k_1
V_1 = X @ W_v_1
Q_2 = X @ W_q_2
K_2 = X @ W_k_2
V_2 = X @ W_v_2

# dot product attention
def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(d_head)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = weights @ V
    return output, weights

# head 1 dot product attention
head_1_output, weights_1 = attention(Q_1, K_1, V_1)
print("head_1_output: \n", head_1_output)
print("weights_1: \n", weights_1)

# head 2 dot product attention
head_2_output, weights_2 = attention(Q_2, K_2, V_2)
print("head_2_output: \n", head_2_output)
print("weights_2: \n", weights_2)

# assert that the attention matrices shape is (sequence_length, d_head)
assert head_1_output.shape == (sequence_length, d_head)
assert head_2_output.shape == (sequence_length, d_head)

# verify that the attention weights in each row sum to approximately 1.0
assert np.allclose(np.sum(weights_1, axis=1), 1.0)
assert np.allclose(np.sum(weights_2, axis=1), 1.0)

# concatenate the attention matrices
attention = np.concatenate([head_1_output, head_2_output], axis=1)
print("attention: \n", attention)

# assert that the attention matrix shape is (sequence_length, d_model)
assert attention.shape == (sequence_length, d_model)

# project the attention matrix into the output matrix
O = attention @ W_o
print("O: \n", O)

# assert that the output matrix shape is (sequence_length, d_model)
assert O.shape == (sequence_length, d_model)

# print the output matrix
print("O: \n", O)


