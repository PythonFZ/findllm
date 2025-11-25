# findllm - Detect AI-generated text by analyzing token prediction patterns using GPT-2.
https://github.com/user-attachments/assets/89d03001-7cef-4b9e-878d-c991bd520f18

## Installation

```bash
# Run directly with uv
uvx findllm document.md

# or install into your environment
pip install findllm
```

## Usage

```bash
# Analyze a file (sentence mode by default)
findllm document.md

# Use token-level analysis
findllm document.md --mode token

# Chunk sentences into smaller batches (e.g., 10 tokens per chunk)
findllm document.md --max-sentence-tokens 10

# Use different aggregation methods (mean, max, l2, rmse, median)
findllm document.md --aggregation max

# JSON output for programmatic use
findllm document.md --json
```

## How It Works

The tool analyzes how predictable each token in the text is according to GPT-2:

- **Green** = Low probability (unpredictable, human-like)
- **Yellow** = Moderate probability
- **Orange** = Higher probability
- **Red** = High probability (predictable, AI-like)

AI-generated text tends to follow predictable patterns that language models can easily anticipate. Human writing is more variable and surprising.

### Metrics

- **Perplexity**: How "surprised" the model is overall (lower = more predictable)
- **Burstiness**: Variation in complexity (humans tend to write with more variation)
- **Color Distribution**: Percentage of green/yellow/orange/red tokens
