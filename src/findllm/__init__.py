import json
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

app = typer.Typer(
    help="Detect AI-generated text by analyzing token prediction patterns"
)
console = Console()


# Document conversion support
try:
    from markitdown import MarkItDown

    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False


def convert_to_markdown(file_path: Path, show_progress: bool = True) -> Optional[str]:
    """Convert a document to markdown using markitdown.

    Args:
        file_path: Path to the file to convert
        show_progress: Whether to show progress indicator

    Returns:
        Markdown text if conversion successful, None otherwise
    """
    if not MARKITDOWN_AVAILABLE:
        return None

    try:
        md = MarkItDown()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(
                    f"Converting {file_path.suffix} to markdown...", total=None
                )
                result = md.convert(str(file_path))
        else:
            result = md.convert(str(file_path))

        return result.markdown
    except Exception as e:
        if show_progress:
            console.print(
                f"[yellow]Warning:[/yellow] Failed to convert {file_path.suffix} file: {e}"
            )
        return None


class AnalysisMode(str, Enum):
    token = "token"
    sentence = "sentence"


class AggregationMethod(str, Enum):
    mean = "mean"
    max = "max"
    l2 = "l2"
    rmse = "rmse"
    median = "median"


def aggregate_probs(probs: list[float], method: AggregationMethod) -> float:
    """Aggregate probabilities using the specified method."""
    if not probs:
        return 0.5

    arr = np.array(probs)

    if method == AggregationMethod.mean:
        return float(np.mean(arr))
    elif method == AggregationMethod.max:
        return float(np.max(arr))
    elif method == AggregationMethod.l2:
        # L2 norm normalized by count (RMS of values)
        return float(np.sqrt(np.mean(arr**2)))
    elif method == AggregationMethod.rmse:
        # RMSE from 0 (same as L2 in this case, but conceptually different)
        return float(np.sqrt(np.mean(arr**2)))
    elif method == AggregationMethod.median:
        return float(np.median(arr))
    else:
        return float(np.mean(arr))


MAX_LENGTH = 1024  # GPT-2 max context window
CHUNK_OVERLAP = 50  # Overlap between chunks for context continuity

SENTENCE_END_PATTERN = re.compile(r"([.!?])(\s+)")


def split_into_sentences(text: str) -> list[tuple[str, str]]:
    """Split text into sentences, preserving trailing whitespace/newlines."""
    results = []
    parts = SENTENCE_END_PATTERN.split(text)

    i = 0
    while i < len(parts):
        sentence = parts[i]
        if i + 2 < len(parts):
            punct = parts[i + 1]
            whitespace = parts[i + 2]
            if sentence.strip():
                results.append((sentence.strip() + punct, whitespace))
            i += 3
        else:
            if sentence.strip():
                results.append((sentence.strip(), ""))
            i += 1

    return results


def get_color_for_prob(
    prob: float,
    threshold_yellow: float,
    threshold_red: float,
    threshold_purple: float,
) -> str:
    """Get color based on probability threshold."""
    if prob > threshold_purple:
        return "bold red"
    elif prob > threshold_red:
        return "rgb(255,165,0)"  # Orange
    elif prob > threshold_yellow:
        return "rgb(255,255,0)"  # Bright yellow
    else:
        return "rgb(0,255,0)"  # Bright green


def calculate_all_token_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    all_tokens: list[int],
) -> list[float]:
    """Calculate probability for each token, handling long sequences with chunking."""
    if len(all_tokens) < 2:
        return []

    probs = []

    # Process in chunks with overlap for long sequences
    for start in range(0, len(all_tokens), MAX_LENGTH - CHUNK_OVERLAP):
        end = min(start + MAX_LENGTH, len(all_tokens))
        chunk_tokens = all_tokens[start:end]

        if len(chunk_tokens) < 2:
            break

        input_ids = torch.tensor([chunk_tokens])

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0].float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            chunk_probs = F.softmax(logits, dim=-1)

        # Determine which tokens to include (skip overlap region except for first chunk)
        token_start = 0 if start == 0 else CHUNK_OVERLAP
        if start == 0:
            token_start = 1  # Skip first token (no prediction for it)

        for i in range(token_start, len(chunk_tokens)):
            token_id = chunk_tokens[i]
            prob = chunk_probs[i - 1, token_id].item()
            probs.append(prob)

        if end >= len(all_tokens):
            break

    return probs


def calculate_perplexity(probs: list[float]) -> float:
    """Calculate perplexity from token probabilities."""
    if not probs:
        return 0.0
    log_probs = [np.log(p + 1e-10) for p in probs]
    avg_log_prob = np.mean(log_probs)
    return float(np.exp(-avg_log_prob))


def calculate_burstiness(probs: list[float], window: int = 10) -> float:
    """Calculate burstiness (variance in local perplexity)."""
    if len(probs) < window:
        return 0.0

    local_perplexities = []
    for i in range(0, len(probs) - window, window // 2):
        chunk = probs[i : i + window]
        local_perplexities.append(calculate_perplexity(chunk))

    if len(local_perplexities) < 2:
        return 0.0
    return float(np.std(local_perplexities))


def get_token_char_spans(
    tokenizer: GPT2TokenizerFast, text: str
) -> list[tuple[int, int, str]]:
    """Get character spans for each token in the text.

    Returns list of (start_char, end_char, token_str) for each token.
    Uses tokenizer's offset mapping for accurate character positions,
    especially important for multi-byte Unicode characters (emojis, special symbols).
    """
    # Use tokenizer's offset mapping for accurate character positions
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    spans = []
    for token_id, (start, end) in zip(tokens, offsets):
        token_str = tokenizer.decode([token_id])
        spans.append((start, end, token_str))

    return spans


def analyze_document(
    text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    mode: AnalysisMode,
    threshold_yellow: float,
    threshold_red: float,
    threshold_purple: float,
    max_sentence_tokens: int = 0,
    min_sentence_tokens: int = 3,
    aggregation_method: AggregationMethod = AggregationMethod.l2,
) -> dict:
    """Analyze document and return results, preserving original line structure."""
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(all_tokens)

    # Calculate all token probabilities for the entire document
    all_probs = calculate_all_token_probs(model, tokenizer, all_tokens)

    # Build character position mapping for tokens
    # Note: all_probs[i] corresponds to all_tokens[i+1] (first token has no probability)
    token_spans = get_token_char_spans(tokenizer, text)
    # token_spans[i] corresponds to all_tokens[i], so for probs we use token_spans[i+1]

    color_counts = {"green": 0, "yellow": 0, "orange": 0, "red": 0}
    lines = []  # List of Text objects, one per original file line

    # Build batches based on mode:
    # - Token mode: 1 token per batch
    # - Sentence mode (max=0): all tokens in sentence per batch
    # - Sentence mode (max=N): chunks of N tokens per batch
    display_probs = all_probs

    # Build list of batches: each batch is (char_start, char_end, token_indices)
    batches: list[tuple[int, int, list[int]]] = []

    if mode == AnalysisMode.token or max_sentence_tokens == 1:
        # One token per batch (starting from token index 1, since 0 has no prob)
        for token_idx in range(1, len(token_spans)):
            char_start, char_end, _ = token_spans[token_idx]
            batches.append(
                (char_start, char_end, [token_idx - 1])
            )  # prob index = token_idx - 1
    else:
        # Sentence mode: group by sentences, then optionally chunk
        original_lines = text.split("\n")
        char_offset = 0

        for original_line in original_lines:
            line_start = char_offset
            line_end = char_offset + len(original_line)

            if original_line.strip():
                line_sentences = split_into_sentences(original_line)
                if not line_sentences:
                    line_sentences = [(original_line, "")]

                sentence_start_in_line = 0
                for sentence, trailing in line_sentences:
                    sentence_start = line_start + sentence_start_in_line
                    sentence_end = sentence_start + len(sentence)

                    # Find token indices for this sentence
                    sentence_token_indices = []
                    for prob_idx in range(len(display_probs)):
                        token_idx = prob_idx + 1
                        if token_idx < len(token_spans):
                            tok_start, tok_end, _ = token_spans[token_idx]
                            if tok_start < sentence_end and tok_end > sentence_start:
                                sentence_token_indices.append(prob_idx)

                    if sentence_token_indices:
                        # Skip sentences that are too short
                        if len(sentence_token_indices) < min_sentence_tokens:
                            continue

                        # Chunk if max_sentence_tokens is set
                        if max_sentence_tokens > 0:
                            for i in range(
                                0, len(sentence_token_indices), max_sentence_tokens
                            ):
                                chunk_indices = sentence_token_indices[
                                    i : i + max_sentence_tokens
                                ]
                                # Get char range for this chunk
                                first_tok = chunk_indices[0] + 1
                                last_tok = chunk_indices[-1] + 1
                                chunk_start = token_spans[first_tok][0]
                                chunk_end = token_spans[last_tok][1]
                                batches.append((chunk_start, chunk_end, chunk_indices))
                        else:
                            batches.append(
                                (sentence_start, sentence_end, sentence_token_indices)
                            )

                    sentence_start_in_line += len(sentence) + len(trailing)

            char_offset = line_end + 1

    # Now render batches to lines, preserving original line structure
    current_line = Text()
    current_char = 0
    line_idx = 0
    original_lines = text.split("\n")
    line_boundaries = []
    pos = 0
    for line in original_lines:
        line_boundaries.append((pos, pos + len(line)))
        pos += len(line) + 1

    # Add first token (no probability) in dim style for token mode
    if mode == AnalysisMode.token and len(token_spans) > 0:
        first_char_start, first_char_end, _ = token_spans[0]
        first_token_text = text[first_char_start:first_char_end]
        if "\n" in first_token_text:
            parts = first_token_text.split("\n")
            for j, part in enumerate(parts):
                if j > 0:
                    lines.append(current_line)
                    current_line = Text()
                current_line.append(part, style="dim")
        else:
            current_line.append(first_token_text, style="dim")
        current_char = first_char_end

    for char_start, char_end, prob_indices in batches:
        # Add any gap text (whitespace between batches) - handle line breaks
        if char_start > current_char:
            gap_text = text[current_char:char_start]
            if "\n" in gap_text:
                parts = gap_text.split("\n")
                for j, part in enumerate(parts):
                    if j > 0:
                        lines.append(current_line)
                        current_line = Text()
                    if part:
                        current_line.append(part)
            elif gap_text:
                current_line.append(gap_text)

        # Aggregate probabilities for this batch
        batch_probs = [display_probs[i] for i in prob_indices]
        agg_prob = aggregate_probs(batch_probs, aggregation_method)

        color = get_color_for_prob(
            agg_prob, threshold_yellow, threshold_red, threshold_purple
        )

        # Count colors
        if "0,255,0" in color:
            color_counts["green"] += 1
        elif "255,255,0" in color:
            color_counts["yellow"] += 1
        elif "255,165,0" in color:
            color_counts["orange"] += 1
        elif "red" in color:
            color_counts["red"] += 1

        # Add batch text with color, handling newlines
        batch_text = text[char_start:char_end]
        if "\n" in batch_text:
            parts = batch_text.split("\n")
            for j, part in enumerate(parts):
                if j > 0:
                    lines.append(current_line)
                    current_line = Text()
                current_line.append(part, style=color)
        else:
            current_line.append(batch_text, style=color)

        current_char = char_end

    # Add any remaining text after last batch
    if current_char < len(text):
        remaining = text[current_char:]
        if "\n" in remaining:
            parts = remaining.split("\n")
            for j, part in enumerate(parts):
                if j > 0:
                    lines.append(current_line)
                    current_line = Text()
                if part:
                    current_line.append(part)
        elif remaining:
            current_line.append(remaining)

    if current_line:
        lines.append(current_line)

    probs_for_metrics = all_probs

    # Calculate metrics
    total_colored = sum(color_counts.values())
    perplexity = calculate_perplexity(probs_for_metrics)
    burstiness = calculate_burstiness(probs_for_metrics)
    avg_prob = float(np.mean(probs_for_metrics)) if probs_for_metrics else 0.0
    high_prob_ratio = (
        sum(1 for p in probs_for_metrics if p > 0.1) / len(probs_for_metrics)
        if probs_for_metrics
        else 0.0
    )

    # Calculate color percentages
    green_pct = color_counts["green"] / total_colored * 100 if total_colored > 0 else 0
    yellow_pct = (
        color_counts["yellow"] / total_colored * 100 if total_colored > 0 else 0
    )
    orange_pct = (
        color_counts["orange"] / total_colored * 100 if total_colored > 0 else 0
    )
    red_pct = color_counts["red"] / total_colored * 100 if total_colored > 0 else 0

    # Determine assessment
    if mode == AnalysisMode.sentence:
        if green_pct >= 60:
            assessment = "Likely human-written"
            ai_score = 1
        elif green_pct >= 40:
            assessment = "Possibly AI-generated"
            ai_score = 2
        else:
            assessment = "Likely AI-generated"
            ai_score = 3
    else:
        ai_score = 0
        if perplexity < 50:
            ai_score += 1
        if burstiness < 20:
            ai_score += 1
        if avg_prob > 0.1:
            ai_score += 1
        if high_prob_ratio > 0.5:
            ai_score += 1

        if ai_score >= 3:
            assessment = "Likely AI-generated"
        elif ai_score >= 2:
            assessment = "Possibly AI-generated"
        else:
            assessment = "Likely human-written"

    return {
        "lines": lines,
        "assessment": assessment,
        "ai_score": ai_score,
        "mode": mode.value,
        "aggregation_method": aggregation_method.value,
        "max_sentence_tokens": max_sentence_tokens,
        "min_sentence_tokens": min_sentence_tokens,
        "total_tokens": total_tokens,
        "metrics": {
            "perplexity": perplexity,
            "burstiness": burstiness,
            "avg_probability": avg_prob,
            "high_prob_ratio": high_prob_ratio,
        },
        "color_distribution": {
            "green_pct": green_pct,
            "yellow_pct": yellow_pct,
            "orange_pct": orange_pct,
            "red_pct": red_pct,
        },
        "thresholds": {
            "yellow": threshold_yellow,
            "red": threshold_red,
            "purple": threshold_purple,
        },
    }


class FindLLMApp(App):
    """Textual TUI for findllm results."""

    CSS = """
    #header-panel {
        height: 5;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 1;
    }

    #legend {
        height: 2;
    }

    #summary {
        height: 2;
    }

    #content {
        height: 1fr;
    }

    #text-view {
        height: 1fr;
        border: solid $primary;
        overflow-y: auto;
        padding: 0 1;
    }

    #search-container {
        height: 3;
        display: none;
        padding: 0 1;
    }

    #search-container.visible {
        display: block;
    }

    #search-input {
        width: 100%;
    }

    .line-number {
        color: $text-muted;
        width: 5;
        text-align: right;
        padding-right: 1;
    }

    Footer {
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "search", "Search"),
        Binding("n", "toggle_lines", "Line Numbers"),
        Binding("m", "toggle_mode", "Mode"),
        Binding("escape", "close_search", "Close Search", show=False),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
        Binding("a", "cycle_aggregation", "Aggregation"),
    ]

    def __init__(
        self,
        results: dict,
        thresholds: dict,
        text: str,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        max_sentence_tokens: int = 0,
        min_sentence_tokens: int = 3,
        aggregation_method: AggregationMethod = AggregationMethod.l2,
    ):
        super().__init__()
        self.results = results
        self.thresholds = thresholds
        self.text = text
        self.model = model
        self.tokenizer = tokenizer
        self.max_sentence_tokens = max_sentence_tokens
        self.min_sentence_tokens = min_sentence_tokens
        self._aggregation_method = aggregation_method
        self.show_line_numbers = False
        self.search_term = ""
        self.lines = results["lines"]
        self._analysis_mode = AnalysisMode(results["mode"])

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        with Vertical(id="header-panel"):
            yield Static(id="legend")
            yield Static(id="summary")

        with Vertical(id="content"):
            with Horizontal(id="search-container"):
                yield Input(
                    placeholder="Search (Enter to confirm, Esc to close)",
                    id="search-input",
                )
            yield RichLog(
                id="text-view",
                highlight=True,
                markup=True,
                wrap=True,
                auto_scroll=False,
            )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the display."""
        self.title = "findllm - AI Text Detection"

        # Build legend
        t = self.thresholds
        legend_text = Text()
        legend_text.append("Legend: ")
        legend_text.append("■ Green", style="rgb(0,255,0)")
        legend_text.append(f" ≤{t['yellow']:.0%}  ")
        legend_text.append("■ Yellow", style="rgb(255,255,0)")
        legend_text.append(f" {t['yellow']:.0%}-{t['red']:.0%}  ")
        legend_text.append("■ Orange", style="rgb(255,165,0)")
        legend_text.append(f" {t['red']:.0%}-{t['purple']:.0%}  ")
        legend_text.append("■ Red", style="bold red")
        legend_text.append(f" >{t['purple']:.0%}")

        self.query_one("#legend", Static).update(legend_text)

        # Build summary
        self._update_summary()

        # Populate text view (auto_scroll=False keeps it at top)
        self._refresh_text_view()

        # Set focus to text view for immediate scrolling
        self.query_one("#text-view", RichLog).focus()

    def _refresh_text_view(self) -> None:
        """Refresh the text view with current settings."""
        text_view = self.query_one("#text-view", RichLog)
        text_view.clear()

        for i, line in enumerate(self.lines, 1):
            display_line = Text()

            if self.show_line_numbers:
                display_line.append(f"{i:4} │ ", style="dim")

            # Apply search highlighting if needed
            if self.search_term and isinstance(line, Text):
                highlighted = self._highlight_search(line)
                display_line.append_text(highlighted)
            else:
                display_line.append_text(line)

            text_view.write(display_line, scroll_end=False)

    def _highlight_search(self, line: Text) -> Text:
        """Highlight search terms in a line."""
        if not self.search_term:
            return line

        plain = line.plain.lower()
        search_lower = self.search_term.lower()

        if search_lower not in plain:
            return line

        # Create new text with highlighting
        result = Text()
        last_end = 0

        idx = 0
        while True:
            pos = plain.find(search_lower, idx)
            if pos == -1:
                break

            # Add text before match (preserving original style)
            if pos > last_end:
                for start, end, style in line._spans:
                    if start < pos and end > last_end:
                        result.append(
                            line.plain[max(start, last_end) : min(end, pos)],
                            style=style,
                        )

            # Add highlighted match
            result.append(
                line.plain[pos : pos + len(self.search_term)], style="black on yellow"
            )

            last_end = pos + len(self.search_term)
            idx = last_end

        # Add remaining text
        if last_end < len(line.plain):
            # Simple append for remaining
            result.append(line.plain[last_end:])

        return result if result.plain else line

    def action_search(self) -> None:
        """Show search input."""
        container = self.query_one("#search-container")
        container.add_class("visible")
        search_input = self.query_one("#search-input", Input)
        search_input.value = self.search_term
        search_input.focus()

    def action_close_search(self) -> None:
        """Hide search input."""
        container = self.query_one("#search-container")
        container.remove_class("visible")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        if event.input.id == "search-input":
            self.search_term = event.value
            self.action_close_search()
            self._refresh_text_view()

    def action_toggle_lines(self) -> None:
        """Toggle line numbers."""
        text_view = self.query_one("#text-view", RichLog)
        scroll_y = text_view.scroll_y
        self.show_line_numbers = not self.show_line_numbers
        self._refresh_text_view()
        self.set_timer(0.1, lambda: text_view.scroll_to(y=scroll_y, animate=False))

    def action_toggle_mode(self) -> None:
        """Toggle between sentence and token mode."""
        # Save scroll position
        text_view = self.query_one("#text-view", RichLog)
        scroll_y = text_view.scroll_y

        # Switch mode
        if self._analysis_mode == AnalysisMode.sentence:
            self._analysis_mode = AnalysisMode.token
        else:
            self._analysis_mode = AnalysisMode.sentence

        # Re-analyze with new mode
        self.results = analyze_document(
            self.text,
            self.model,
            self.tokenizer,
            self._analysis_mode,
            self.thresholds["yellow"],
            self.thresholds["red"],
            self.thresholds["purple"],
            self.max_sentence_tokens,
            self.min_sentence_tokens,
            self._aggregation_method,
        )
        self.lines = self.results["lines"]

        # Update summary
        self._update_summary()
        self._refresh_text_view()

        # Restore scroll position after content is rendered (use timer to ensure rendering is complete)
        self.set_timer(0.1, lambda: text_view.scroll_to(y=scroll_y, animate=False))

    def action_cycle_aggregation(self) -> None:
        """Cycle through aggregation methods (only affects sentence mode)."""
        if self._analysis_mode != AnalysisMode.sentence:
            return  # Aggregation only matters in sentence mode

        # Save scroll position
        text_view = self.query_one("#text-view", RichLog)
        scroll_y = text_view.scroll_y

        # Cycle to next aggregation method
        methods = list(AggregationMethod)
        current_idx = methods.index(self._aggregation_method)
        self._aggregation_method = methods[(current_idx + 1) % len(methods)]

        # Re-analyze with new aggregation method
        self.results = analyze_document(
            self.text,
            self.model,
            self.tokenizer,
            self._analysis_mode,
            self.thresholds["yellow"],
            self.thresholds["red"],
            self.thresholds["purple"],
            self.max_sentence_tokens,
            self.min_sentence_tokens,
            self._aggregation_method,
        )
        self.lines = self.results["lines"]

        # Update summary
        self._update_summary()
        self._refresh_text_view()

        # Restore scroll position
        self.set_timer(0.1, lambda: text_view.scroll_to(y=scroll_y, animate=False))

    def _update_summary(self) -> None:
        """Update the summary display."""
        r = self.results
        cd = r["color_distribution"]
        summary_text = Text()

        # Assessment with color
        assessment = r["assessment"]
        if "human" in assessment.lower():
            summary_text.append(f"✓ {assessment}", style="bold green")
        elif "Possibly" in assessment:
            summary_text.append(f"? {assessment}", style="bold yellow")
        else:
            summary_text.append(f"✗ {assessment}", style="bold red")

        # Show current mode and aggregation (only in sentence mode)
        mode_str = (
            "sentence" if self._analysis_mode == AnalysisMode.sentence else "token"
        )
        if self._analysis_mode == AnalysisMode.sentence:
            agg_str = f"  │  Agg: {self._aggregation_method.value}"
        else:
            agg_str = ""
        summary_text.append(
            f"  │  Mode: {mode_str}{agg_str}  │  Tokens: {r['total_tokens']}  │  "
        )
        summary_text.append(f"{cd['green_pct']:.0f}%", style="rgb(0,255,0)")
        summary_text.append("/")
        summary_text.append(f"{cd['yellow_pct']:.0f}%", style="rgb(255,255,0)")
        summary_text.append("/")
        summary_text.append(f"{cd['orange_pct']:.0f}%", style="rgb(255,165,0)")
        summary_text.append("/")
        summary_text.append(f"{cd['red_pct']:.0f}%", style="bold red")

        self.query_one("#summary", Static).update(summary_text)

    def action_scroll_down(self) -> None:
        """Scroll down."""
        self.query_one("#text-view", RichLog).scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll up."""
        self.query_one("#text-view", RichLog).scroll_up()

    def action_scroll_home(self) -> None:
        """Scroll to top."""
        self.query_one("#text-view", RichLog).scroll_home()

    def action_scroll_end(self) -> None:
        """Scroll to bottom."""
        self.query_one("#text-view", RichLog).scroll_end()


def run_analysis(
    text: str,
    mode: AnalysisMode,
    threshold_yellow: float,
    threshold_red: float,
    threshold_purple: float,
    json_output: bool,
    max_sentence_tokens: int = 0,
    min_sentence_tokens: int = 3,
    aggregation_method: AggregationMethod = AggregationMethod.l2,
) -> tuple[dict, GPT2LMHeadModel, GPT2TokenizerFast]:
    """Run the analysis and return results along with model/tokenizer for TUI."""
    # Load model
    if json_output:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e30)
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading GPT-2 model...", total=None)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.model_max_length = int(1e30)
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.eval()

    # Analyze document
    if not json_output:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Analyzing document...", total=None)
            results = analyze_document(
                text,
                model,
                tokenizer,
                mode,
                threshold_yellow,
                threshold_red,
                threshold_purple,
                max_sentence_tokens,
                min_sentence_tokens,
                aggregation_method,
            )
    else:
        results = analyze_document(
            text,
            model,
            tokenizer,
            mode,
            threshold_yellow,
            threshold_red,
            threshold_purple,
            max_sentence_tokens,
            min_sentence_tokens,
            aggregation_method,
        )

    return results, model, tokenizer


@app.command()
def main(
    file: Annotated[Path, typer.Argument(help="Path to the text file to analyze")],
    mode: Annotated[
        AnalysisMode,
        typer.Option("--mode", "-m", help="Analysis mode: token or sentence"),
    ] = AnalysisMode.sentence,
    threshold_yellow: Annotated[
        float,
        typer.Option("--threshold-yellow", help="Probability threshold for yellow"),
    ] = 0.38,
    threshold_red: Annotated[
        float, typer.Option("--threshold-red", help="Probability threshold for red")
    ] = 0.45,
    threshold_purple: Annotated[
        float,
        typer.Option("--threshold-purple", help="Probability threshold for purple"),
    ] = 0.55,
    json_output: Annotated[
        bool, typer.Option("--json", "-j", help="Output results as JSON")
    ] = False,
    max_sentence_tokens: Annotated[
        int,
        typer.Option(
            "--max-sentence-tokens",
            "-c",
            help="Max tokens per chunk (0=full sentence/token)",
        ),
    ] = 0,
    min_sentence_tokens: Annotated[
        int,
        typer.Option(
            "--min-sentence-tokens",
            help="Min tokens per sentence (sentences with fewer tokens are skipped)",
        ),
    ] = 3,
    aggregation_method: Annotated[
        AggregationMethod,
        typer.Option(
            "--aggregation",
            "-a",
            help="Aggregation method: mean, max, l2, rmse, median",
        ),
    ] = AggregationMethod.l2,
) -> None:
    """Analyze a text file for AI-generated content using GPT-2 prediction patterns.

    Supports plain text files and various document formats (PDF, DOCX, PPTX, etc.)
    when installed with the 'markitdown' extra: pip install findllm[markitdown]
    """
    if not file.exists():
        if json_output:
            print(json.dumps({"error": f"File '{file}' not found."}))
        else:
            console.print(f"[red]Error:[/red] File '{file}' not found.")
        raise typer.Exit(1)

    # Try markitdown conversion for non-.md files if available
    text = None
    if file.suffix.lower() != ".md" and MARKITDOWN_AVAILABLE:
        text = convert_to_markdown(file, show_progress=not json_output)

    # Fall back to reading as plain text if conversion failed or not attempted
    if text is None:
        try:
            text = file.read_text()
        except Exception as e:
            error_msg = f"Failed to read '{file}': {e}"
            if json_output:
                print(json.dumps({"error": error_msg}))
            else:
                console.print(f"[red]Error:[/red] {error_msg}")
            raise typer.Exit(1)

    if not text.strip():
        if json_output:
            print(json.dumps({"error": "File is empty."}))
        else:
            console.print("[red]Error:[/red] File is empty.")
        raise typer.Exit(1)

    results, model, tokenizer = run_analysis(
        text,
        mode,
        threshold_yellow,
        threshold_red,
        threshold_purple,
        json_output,
        max_sentence_tokens,
        min_sentence_tokens,
        aggregation_method,
    )

    if json_output:
        # Remove lines from JSON output (not serializable as-is)
        output = {k: v for k, v in results.items() if k != "lines"}
        print(json.dumps(output, indent=2))
    else:
        # Launch TUI
        tui = FindLLMApp(
            results,
            {
                "yellow": threshold_yellow,
                "red": threshold_red,
                "purple": threshold_purple,
            },
            text,
            model,
            tokenizer,
            max_sentence_tokens,
            min_sentence_tokens,
            aggregation_method,
        )
        tui.run()


if __name__ == "__main__":
    app()
