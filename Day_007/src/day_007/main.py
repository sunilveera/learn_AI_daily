import numpy as np

# sinusoidal positional encoding
def sinusoidal_positional_encoding(sequence_length, embedding_dimension):
    # create a matrix of positions and dimensions
    position = np.arange(sequence_length)[:, np.newaxis]
    dimension = np.arange(embedding_dimension)[np.newaxis, :]

    # create a matrix of angle rates
    angle_rates = 1 / np.power(10000, (2 * (dimension // 2)) / embedding_dimension)
    
    # create a matrix of angles
    angle = position * angle_rates

    # create positional encoding matrix
    positional_encoding = np.zeros((sequence_length, embedding_dimension))

    # fill the even dimensions with sine angles
    positional_encoding[:, 0::2] = np.sin(angle[:, 0::2])

    # fill the odd dimensions with cosine angles
    positional_encoding[:, 1::2] = np.cos(angle[:, 1::2])
    
    return positional_encoding



# token embedding for the sequence "very very good"
token_embedding = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # very -> token embedding 1
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # very -> token embedding 2
    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], # good -> token embedding 3
     ])

# positional encoding for the sequence
positional_encoding = sinusoidal_positional_encoding(3, 8)

# print the positional encoding
print("Positional Encoding:\n")
print(positional_encoding)

# sum the token embedding and the positional encoding
token_embedding_plus_positional_encoding = token_embedding + positional_encoding

# print the token embedding plus positional encoding
print("\n\nToken Embedding + Positional Encoding:\n")
print(token_embedding_plus_positional_encoding)

assert token_embedding.shape == positional_encoding.shape == \
    token_embedding_plus_positional_encoding.shape == (3, 8)

assert np.allclose(token_embedding[0], token_embedding[1])

assert not np.allclose(
    token_embedding_plus_positional_encoding[0],
    token_embedding_plus_positional_encoding[1],
)

# print the token embedding plus positional encoding
print("\n\nToken Embedding + Positional Encoding (rounded to 3 decimal places while maintaining the original precision):\n")
print(np.round(token_embedding_plus_positional_encoding, 3))
