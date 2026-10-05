from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


prompt = input("Enter a prompt: ")

generated_text = prompt

for _ in range(10):
    # tokenizer the prompt
    inputs = tokenizer(generated_text, return_tensors="pt")

    # pass the inputs to the model
    with torch.no_grad():
        outputs = model(**inputs)

    # last token of the logits
    next_token_logits = outputs.logits[0, -1, :]

    # convert the logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # find the most likely next token
    next_token_id = torch.argmax(probabilities)

    # decode the next token
    next_token = tokenizer.decode(next_token_id)

    generated_text += next_token

    print(generated_text)
