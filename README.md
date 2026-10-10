# learn_AI_daily

Daily notes and small programs for learning how language models work, from tokenization through a transformer block to generating text.

Each day is its own folder and its own [uv](https://docs.astral.sh/uv/) project. Python 3.12 or newer. From that day's folder, install with `uv sync`, then run the script shown in that day's README.

| Day | Topic | Start here |
| --- | --- | --- |
| [Day 001](Day_001/README.md) | Walk a sentence through the `bert-base-uncased` tokenizer: tokens, IDs, a PyTorch encoding, and back to text. | `uv run python src/day_001/main.py` |
| [Day 002](Day_002/README.md) | Embed two sentences with `all-MiniLM-L6-v2` and score them with cosine similarity. | `uv run python src/day_002/main.py` |
| [Day 003](Day_003/README.md) | Scaled dot-product attention in NumPy on a fixed query, key, and value. | `uv run python src/day_003/main.py` |
| [Day 004](Day_004/README.md) | One transformer block: attention, residual connections, layer norm, and a small feed-forward network. | `uv run python src/day_004/main.py` |
| [Day 005](Day_005/README.md) | Next-token prediction with `distilgpt2`: the top 5 candidates, then a 10-token greedy continuation. | `uv run python src/day_005/top5.py` |
| [Day 006](Day_006/README.md) | Sample the next token with temperature, top-k, and top-p, then compare three settings on one prompt. | `uv run python src/day_006/main.py` |
| [Day 007](Day_007/README.md) | Add sinusoidal positional encodings to a token embedding so identical tokens at different positions stay distinct. | `uv run python src/day_007/main.py` |
| [Day 008](Day_008/README.md) | Run two attention heads, concatenate them, and project the result back to the model width. | `uv run python src/day_008/main.py` |
| [Day 009](Day_009/README.md) | Compare encoder self-attention, causal decoder attention, and cross-attention from decoder queries onto encoder keys. | `uv run python src/day_009/main.py` |
| [Day 010](Day_010/README.md) | Run one untrained decoder block on `the cat` and print a next-token distribution over a 10-word vocabulary. | `uv run python src/day_010/main.py` |

Days 001, 002, 005, and 006 download a Hugging Face model or tokenizer the first time they run.

This project is licensed under the GNU General Public License v3. See [LICENSE](LICENSE).
