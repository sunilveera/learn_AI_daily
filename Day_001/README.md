# Day 001 — BERT tokenizer walkthrough

`main.py` loads the `bert-base-uncased` tokenizer and walks one line of text through the usual Hugging Face steps: split into tokens, map those tokens to vocabulary IDs, encode the full sentence as a PyTorch tensor (including special tokens), decode that tensor back to text, then turn the IDs back into tokens.

## What it does

1. Loads `AutoTokenizer` for `bert-base-uncased` (downloads the tokenizer files on first run).
2. Reads a line of text from the terminal.
3. Prints WordPiece tokens from `tokenizer.tokenize`.
4. Prints vocabulary IDs from `tokenizer.convert_tokens_to_ids`.
5. Prints the model-ready tensor from `tokenizer.encode` (`return_tensors="pt"`). This adds `[CLS]` and `[SEP]`.
6. Prints the decoded string from `tokenizer.decode`.
7. Prints tokens again from `tokenizer.convert_ids_to_tokens`.

`tokenize` does not add special tokens. `encode` does, so the decoded line is wrapped as `[CLS] ... [SEP]`.

## Requirements

- Python 3.10+
- Packages in `requirements.txt`: [transformers](https://huggingface.co/docs/transformers) with the `torch` extra, which installs [PyTorch](https://pytorch.org/) (`encode` returns a `torch.Tensor`)

From this folder:

```bash
uv pip install -r requirements.txt
```

## Run

From this folder:

```bash
python main.py
```

Example:

```text
Enter a text: Hello, world!
['hello', ',', 'world', '!']
[7592, 1010, 2088, 999]
tensor([[ 101, 7592, 1010, 2088, 999, 102]])
[CLS] hello, world! [SEP]
['hello', ',', 'world', '!']
```

`101` is `[CLS]` and `102` is `[SEP]`. Those IDs appear only in the encoded tensor.
