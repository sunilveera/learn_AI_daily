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

Days 001, 002, and 005 download a Hugging Face model or tokenizer the first time they run.

This project is licensed under the GNU General Public License v3. See [LICENSE](LICENSE).
