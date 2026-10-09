import numpy as np

# number of tokens in the sequence
source_length = 5
# number of tokens in the target sequence
target_length = 3
# dimension of the model
d_model = 8
# number of heads in the multi-head attention
num_heads = 2
# dimension of each head
d_head = d_model // num_heads

# using a seed for reproducibility
np.random.seed(42)


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

# create a matrix of token embeddings for encoder
X = np.random.randn(source_length, d_model)
print("X: \n", X)

# projection matrices for head 1
W_q_encoder_1 = np.random.randn(d_model, d_head)
W_k_encoder_1 = np.random.randn(d_model, d_head)
W_v_encoder_1 = np.random.randn(d_model, d_head)

# Simulate encoder-only architecture: no causal mask needed
# project the token embeddings into the query, key, and value vectors
Q_encoder_1 = X @ W_q_encoder_1
K_encoder_1 = X @ W_k_encoder_1
V_encoder_1 = X @ W_v_encoder_1

# dot product attention for encoder
encoder_weights_1, encoder_head_1_output  = attention(Q_encoder_1, K_encoder_1, V_encoder_1)
print("encoder head_1_output: \n", encoder_head_1_output)
print("encoder weights_1: \n", encoder_weights_1)

# assert Attention weights: (5, 5)
assert encoder_weights_1.shape == (source_length, source_length)
# Encoder output: (5, 4)
assert encoder_head_1_output.shape == (source_length, d_head)
# Every attention-weight row sums approximately to 1.0.
assert np.allclose(np.sum(encoder_weights_1, axis=-1), 1.0)


# create a matrix of token embeddings for decoder
Y = np.random.randn(target_length, d_model)
print("Y: \n", Y)

# projection matrices for head 1
W_q_decoder_1 = np.random.randn(d_model, d_head)
W_k_decoder_1 = np.random.randn(d_model, d_head)
W_v_decoder_1 = np.random.randn(d_model, d_head)

# project the token embeddings into the query, key, and value vectors
Q_decoder_1 = Y @ W_q_decoder_1
K_decoder_1 = Y @ W_k_decoder_1
V_decoder_1 = Y @ W_v_decoder_1

# dot product attention for decoder
# apply causal mask to the decoder attention
decoder_mask = np.triu(np.ones((target_length, target_length), dtype=bool), k=1)
decoder_weights_1, decoder_head_1_output  = attention(Q_decoder_1, K_decoder_1, V_decoder_1, mask=decoder_mask)
print("decoder head_1_output: \n", decoder_head_1_output)
print("decoder weights_1: \n", decoder_weights_1)

# assert Attention weights: (3, 3)
assert decoder_weights_1.shape == (target_length, target_length)
# Decoder output: (3, 4)
assert decoder_head_1_output.shape == (target_length, d_head)
# Every attention-weight row sums approximately to 1.0.
assert np.allclose(np.sum(decoder_weights_1, axis=-1), 1.0)
assert np.allclose(np.triu(decoder_weights_1, k=1), 0.0)

# Implement cross-attention using:
# Q: projected decoder output.
W_q_cross_attention = np.random.randn(d_head, d_head)
# K: projected encoder output.
W_k_cross_attention = np.random.randn(d_head, d_head)
# V: projected encoder output.
W_v_cross_attention = np.random.randn(d_head, d_head)
Q_cross_attention = decoder_head_1_output @ W_q_cross_attention
K_cross_attention = encoder_head_1_output @ W_k_cross_attention
V_cross_attention = encoder_head_1_output @ W_v_cross_attention

# dot product attention for cross-attention
cross_attention_weights_1, cross_attention_head_1_output  = attention(Q_cross_attention, K_cross_attention, V_cross_attention)
print("cross-attention head_1_output: \n", cross_attention_head_1_output)
print("cross-attention weights_1: \n", cross_attention_weights_1)

# assert Cross-attention weights: (3, 5)
assert cross_attention_weights_1.shape == (target_length, source_length)
# Cross-attention output: (3, 4)
assert cross_attention_head_1_output.shape == (target_length, d_head)
# Every attention-weight row sums approximately to 1.0.
assert np.allclose(np.sum(cross_attention_weights_1, axis=-1), 1.0)