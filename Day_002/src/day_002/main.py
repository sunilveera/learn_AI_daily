from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# get text input from user
text1 = input("Enter the first text: ")
text2 = input("Enter the second text: ")

# get embedding of the text
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)

# print the embedding dimensions
print("embedding shape: ", embedding1.shape)

# print the first 10 elements of the embedding
print("First 10 elements of embedding1: ", embedding1[:10])
print("First 10 elements of embedding2: ", embedding2[:10])

# calculate the similarity between the two embeddings
similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print("Similarity: ", similarity)