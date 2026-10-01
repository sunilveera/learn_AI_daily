from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# take input from user
input_text = input("Enter a text: ")

# convert user input to tokens
tokens = tokenizer.tokenize(input_text)
print(tokens)

# # convert tokens to ids
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

encoded_input = tokenizer.encode(input_text, return_tensors="pt")
print(encoded_input)

decoded_input = tokenizer.decode(encoded_input[0])
print(decoded_input)

# # convert ids to tokens
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)

# convert ids to tokens