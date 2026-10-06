from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


prompt = input("Enter a prompt: ")

generated_text = prompt

# apply top k to the logits
def apply_top_k(logits, k=5):
    values, indices = torch.topk(logits, min(k, logits.size(-1)))
    filtered = torch.full_like(logits, -float("inf"))
    filtered.scatter_(-1, indices, values)
    return filtered

# apply top p to the logits
def apply_top_p(logits, p=0.9):
    # Rank tokens from most likely to least likely.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Drop tokens once the running total has already passed p.
    # Shift the mask right so the token that crosses p is kept.
    past_cutoff = cumulative_probs > p
    remove_when_sorted = torch.zeros_like(past_cutoff)
    remove_when_sorted[..., 1:] = past_cutoff[..., :-1]

    # Map that mask back onto the original vocabulary order.
    remove = torch.zeros_like(logits, dtype=torch.bool)
    remove.scatter_(0, sorted_indices, remove_when_sorted)

    filtered = logits.clone()
    filtered[remove] = -float("inf")
    return filtered

def generate_text(prompt, num_tokens=20, temperature=0.8, top_k=20, top_p=0.8):
    for _ in range(num_tokens):
        # tokenizer the prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # pass the inputs to the model
        with torch.no_grad():
            outputs = model(**inputs)

        # last token of the logits divided by temperature
        next_token_logits = outputs.logits[0, -1, :] / temperature

        # apply top k to the logits
        next_token_logits = apply_top_k(next_token_logits, top_k)

        # apply top p to the logits
        next_token_logits = apply_top_p(next_token_logits, top_p)

        # convert the logits to probabilities
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # find the next token using multinomial sampling
        # argmax → deterministic, multinomial → sampling
        next_token_id = torch.multinomial(probabilities, num_samples=1)

        # decode the next token
        next_token = tokenizer.decode(next_token_id)

        prompt += next_token

        print(prompt)
        
    return prompt

print("Experiment A: Temperature = 0.3, Top K = 20, Top P = 0.8")
generated_text = generate_text(generated_text, num_tokens=20, temperature=0.3, top_k=20, top_p=0.8)
print(generated_text)

print("Experiment B: Temperature = 1.0, Top K = 40, Top P = 0.9")
generated_text = generate_text(generated_text, num_tokens=20, temperature=1.0, top_k=40, top_p=0.9)
print(generated_text)

print("Experiment C: Temperature = 1.5, Top K = 100, Top P = 0.95")
generated_text = generate_text(generated_text, num_tokens=20, temperature=1.5, top_k=100, top_p=0.95)
print(generated_text)