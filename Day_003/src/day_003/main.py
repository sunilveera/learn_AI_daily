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

scores = Q @ K.T
print("Scores: \n", scores)

scale_factor = np.sqrt(K.shape[-1])

norm_scores = scores / scale_factor
print("Normalized Scores: \n", norm_scores)

# softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

attention_weights = softmax(norm_scores)
print("Attention Weights: \n", attention_weights)

output = attention_weights @ V
print("Output: \n", output)
