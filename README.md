# findllm - Detect AI-generated text by analyzing token prediction patterns using GPT-2.
https://github.com/user-attachments/assets/89d03001-7cef-4b9e-878d-c991bd520f18

## Installation

> [!IMPORTANT]
> This project was 99% vibe coded using gemini 3 Pro, sonnet 4.5 and opus 4.5.
> The model tends to identify AI text as human text more often than the reverse. 
> The metrics can't be fully trusted though, so use everything with caution!

```bash
# Run directly with uv
uvx findllm document.md

# With document conversion support (PDF, DOCX, PPTX, etc.)
uvx 'findllm[markitdown]' document.pdf

# Or install into your environment
pip install findllm

# With document conversion support
pip install 'findllm[markitdown]'
```

## Usage

```bash
# Analyze a text file (sentence mode by default)
findllm document.md

# Analyze documents (PDF, DOCX, PPTX, XLSX, etc.) - requires markitdown extra
findllm paper.pdf
findllm presentation.pptx
findllm report.docx

# Use token-level analysis
findllm document.md --mode token

# Chunk sentences into smaller batches (e.g., 10 tokens per chunk)
findllm document.md --max-sentence-tokens 10

# Skip short sentences (e.g., less than 5 tokens, default is 3)
findllm document.md --min-sentence-tokens 5

# Use different aggregation methods (mean, max, l2, rmse, median, default is l2)
findllm document.md --aggregation max

# JSON output for programmatic use
findllm document.md --json
```

## Document Support

With the `markitdown` extra installed, findllm can analyze various document formats:

- **PDF** (.pdf)
- **Microsoft Office** (.docx, .pptx, .xlsx)
- **HTML** (.html, .htm)
- **Rich Text** (.rtf)
- **And more** - see [markitdown](https://github.com/microsoft/markitdown) for full format support

Documents are automatically converted to markdown in-memory before analysis.

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

### Evaluation
Evaluation on 150 samples from https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

<img width="223" height="176" alt="Image" src="https://github.com/user-attachments/assets/ffe96501-548f-4868-90d0-f8b33d55619d" />
