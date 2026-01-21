#!/usr/bin/env python3
"""
Export BERT model to ONNX format for use with onnx_zig.

This script downloads bert-base-uncased from Hugging Face and exports it to ONNX.
It also saves the vocabulary file needed for tokenization.

Usage:
    uv run --with transformers --with torch --with onnx scripts/export_bert.py

Output:
    models/bert/bert-base-uncased.onnx  - The ONNX model
    models/bert/vocab.txt               - The vocabulary file
    models/bert/config.json             - Model configuration
"""

import os
import json
import shutil
from pathlib import Path


def main():
    # Import here to allow script to show help without dependencies
    import torch
    from transformers import BertModel, BertTokenizer, BertConfig

    output_dir = Path("models/bert")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BERT model and tokenizer...")
    model_name = "google-bert/bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)

    model.eval()

    # Save vocabulary
    vocab_path = output_dir / "vocab.txt"
    tokenizer.save_vocabulary(str(output_dir))
    print(f"Saved vocabulary to {vocab_path}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": config.max_position_embeddings,
            "type_vocab_size": config.type_vocab_size,
        }, f, indent=2)
    print(f"Saved config to {config_path}")

    # Create dummy inputs for export
    batch_size = 1
    seq_length = 128

    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)

    # Export to ONNX
    onnx_path = output_dir / "bert-base-uncased.onnx"
    print(f"Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "token_type_ids": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"},
            "pooler_output": {0: "batch_size"},
        },
        dynamo=False,  # Use legacy exporter for compatibility
    )

    # Verify the model
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")

    # Print model info
    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nModel exported successfully!")
    print(f"  Path: {onnx_path}")
    print(f"  Size: {model_size:.1f} MB")
    print(f"  Inputs: input_ids, attention_mask, token_type_ids")
    print(f"  Outputs: last_hidden_state [batch, seq, 768], pooler_output [batch, 768]")

    # Test tokenization
    print("\nTest tokenization...")
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=seq_length, truncation=True)
    print(f"  Input: '{text}'")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:10])}...")
    print(f"  Token IDs: {inputs['input_ids'][0][:10].tolist()}")
    print("\nReady for inference with onnx_zig!")


if __name__ == "__main__":
    main()
