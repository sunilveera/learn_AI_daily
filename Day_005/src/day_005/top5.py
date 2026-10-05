from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


prompt = input("Enter a prompt: ")

# tokenizer the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# print the input ids
print(inputs["input_ids"])

# pass the inputs to the model
with torch.no_grad():
    outputs = model(**inputs)

# print the outputs
print(outputs.logits.shape)

# last token of the logits
next_token_logits = outputs.logits[0, -1, :]

# convert the logits to probabilities
probabilities = torch.softmax(next_token_logits, dim=-1)

# find the top 5 most likely next tokens
top_5_tokens = torch.topk(probabilities, 5)

# print the top 5 most likely next tokens with their probabilities in a table format 
print("Top 5 most likely next tokens with their probabilities:")
print("--------------------------------")
print("| Token | Probability |")
print("--------------------------------")
for token, probability in zip(top_5_tokens.indices, top_5_tokens.values):
    # decode the token
    token_str = tokenizer.decode(token.item())
    # add padding to the token string to make it 10 characters long
    token_str = token_str.ljust(10)
    # add padding to the probability to make it 10 characters long
    probability_str = str(probability.item()).ljust(10, '0')
    print(f"| {token_str} | {probability_str} |")
print("--------------------------------")
