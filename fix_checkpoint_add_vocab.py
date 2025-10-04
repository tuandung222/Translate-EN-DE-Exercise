"""
Script to add vocabulary to existing checkpoints.
This is needed for checkpoints trained before vocab was saved.
"""

import torch
import argparse
from data_loader import prepareData


def add_vocab_to_checkpoint(checkpoint_path, output_path=None):
    """Add vocabulary to an existing checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "input_lang" in checkpoint and "output_lang" in checkpoint:
        print("✓ Checkpoint already has vocabulary!")
        return

    print("⚠ Vocabulary not found in checkpoint, creating...")

    # Create vocabulary using the SAME random seed as training
    input_lang, output_lang, _, _, _ = prepareData(
        data_source="local",
        lang1="ger",
        lang2="eng",
        reverse=False,
        max_length=64,
        batch_size=32,
    )

    # Verify vocab sizes match checkpoint
    encoder_vocab_size = checkpoint["encoder_state_dict"]["embedding.weight"].shape[0]
    decoder_vocab_size = checkpoint["decoder_state_dict"]["embedding.weight"].shape[0]

    print(f"Checkpoint vocab sizes - Encoder: {encoder_vocab_size}, Decoder: {decoder_vocab_size}")
    print(f"Created vocab sizes - Input: {input_lang.n_words}, Output: {output_lang.n_words}")

    if input_lang.n_words != encoder_vocab_size or output_lang.n_words != decoder_vocab_size:
        print("❌ ERROR: Vocabulary size mismatch!")
        print("   This means the data loading produces different vocab each time.")
        print("   You need to re-train the model with the fixed train.py that saves vocab.")
        return False

    # Add vocabulary to checkpoint
    checkpoint["input_lang"] = input_lang
    checkpoint["output_lang"] = output_lang

    if output_path is None:
        output_path = checkpoint_path

    print(f"Saving updated checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("✓ Vocabulary added successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add vocabulary to checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_attention/best_model.pth",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrite input)",
    )

    args = parser.parse_args()
    add_vocab_to_checkpoint(args.checkpoint, args.output)
