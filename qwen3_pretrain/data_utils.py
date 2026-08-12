"""Data utilities for pretraining Qwen3-0.6B from scratch on alpaca_zh."""

# SPDX-License-Identifier: Apache-2.0


def format_example(example):
    """Render one alpaca_zh record as plain text for causal-LM pretraining."""
    instruction = example["instruction"]
    input_text = example.get("input") or ""
    output = example["output"]
    parts = [instruction]
    if input_text:
        parts.append(input_text)
    parts.append(output)
    return "\n\n".join(parts)


def pack_sequences(tokenizer, texts, seq_len, eos_token_id):
    """Tokenize all texts, concatenate with eos separators, cut into windows."""
    tokens = []
    for i, text in enumerate(texts):
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if i < len(texts) - 1:
            tokens.append(eos_token_id)
    return [tokens[i:i + seq_len] for i in range(0, len(tokens) - seq_len + 1, seq_len)]
