# llmcheck

Detect AI-generated text by analyzing token prediction patterns using GPT-2.

## Installation

```bash
# Run directly with uv
uvx llmcheck document.md

# or install into your environment
pip install llmcheck
```

## Usage

```bash
# Analyze a file (sentence mode by default)
llmcheck document.md

# Use token-level analysis with smoothing
llmcheck document.md --mode token

# Adjust smoothing window (token mode)
llmcheck document.md --mode token --window 10
```

## How It Works

The tool analyzes how predictable each token in the text is according to GPT-2:

- **Green** = Low probability (unpredictable, human-like)
- **Yellow** = Moderate probability
- **Red** = High probability (predictable, AI-like)

AI-generated text tends to follow predictable patterns that language models can easily anticipate. Human writing is more variable and surprising.

### Metrics

- **Perplexity**: How "surprised" the model is overall (lower = more predictable)
- **Burstiness**: Variation in complexity (humans tend to write with more variation)
- **Color Distribution**: Percentage of green/yellow/red tokens
