"""
Inference script for translation with nucleus sampling.
"""

import argparse
import logging
import torch
import torch.nn.functional as F

from data_loader import prepareData
from model import (
    StackedGRUEncoder,
    StackedGRUDecoder,
    StackedGRUAttnDecoder,
    ResidualStackedGRUEncoder,
    ResidualStackedGRUDecoder,
    ResidualStackedGRUAttnDecoder,
)
from utils import get_device, SOS_token, EOS_token

try:
    from huggingface_hub import hf_hub_download

    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def nucleus_sampling(
    logits, top_k=20, top_p=0.6, temperature=0.3, repetition_penalty=1.05, generated_tokens=None
):
    """
    Apply nucleus (top-p) sampling with top-k filtering and repetition penalty.

    Args:
        logits: (vocab_size,) tensor of logits
        top_k: keep only top-k highest probability tokens
        top_p: cumulative probability threshold for nucleus sampling
        temperature: sampling temperature (lower = more deterministic)
        repetition_penalty: penalty for repeating tokens (>1.0 = penalize repetition)
        generated_tokens: list of previously generated token IDs for repetition penalty

    Returns:
        sampled token index
    """
    # logits: (vocab_size,)

    # Apply repetition penalty to previously generated tokens
    if generated_tokens is not None and len(generated_tokens) > 0 and repetition_penalty != 1.0:
        for token_id in set(generated_tokens):
            if logits[token_id] < 0:
                logits[token_id] *= repetition_penalty  # (vocab_size,)
            else:
                logits[token_id] /= repetition_penalty  # (vocab_size,)

    # Apply temperature scaling
    logits = logits / temperature  # (vocab_size,)

    # Top-k filtering: keep only top k tokens
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(
            logits, min(top_k, logits.size(-1))
        )  # top_k_logits: (top_k,), top_k_indices: (top_k,)
        logits_filtered = torch.full_like(logits, float("-inf"))  # (vocab_size,)
        logits_filtered.scatter_(0, top_k_indices, top_k_logits)  # (vocab_size,)
        logits = logits_filtered  # (vocab_size,)

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # (vocab_size,)

    # Top-p (nucleus) filtering
    sorted_probs, sorted_indices = torch.sort(
        probs, descending=True
    )  # sorted_probs: (vocab_size,), sorted_indices: (vocab_size,)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (vocab_size,)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p  # (vocab_size,) boolean
    # Shift the indices to the right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # (vocab_size,)
    sorted_indices_to_remove[..., 0] = 0  # Always keep the top token

    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )  # (vocab_size,)
    probs[indices_to_remove] = 0  # (vocab_size,)

    # Renormalize
    if probs.sum() > 0:
        probs = probs / probs.sum()  # (vocab_size,)
    else:
        # Fallback: if all probabilities are zero, use uniform distribution
        probs = torch.ones_like(probs) / probs.size(0)  # (vocab_size,)

    # Sample from the filtered distribution
    next_token = torch.multinomial(probs, num_samples=1)  # (1,)
    return next_token.item()  # int


def translate_sentence(
    encoder,
    decoder,
    sentence,
    input_lang,
    output_lang,
    device,
    max_length=64,
    use_nucleus=True,
    top_k=20,
    top_p=0.6,
    temperature=0.3,
    repetition_penalty=1.05,
):
    """Translate a single sentence with nucleus sampling."""
    encoder.eval()
    decoder.eval()

    words = sentence.split()
    indexes = [input_lang.word2index.get(word, 0) for word in words]
    indexes.append(EOS_token)
    input_tensor = torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        # Input tensors remain as Long for embedding layers
        encoder_hidden = encoder.initHidden(1, device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_input = torch.full((1, 1), SOS_token, device=device, dtype=torch.long)
        decoder_hidden = encoder_hidden

        decoded_words = []
        generated_token_ids = []

        for _ in range(max_length):
            decoder_output, decoder_hidden, attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            logits = decoder_output[0, -1, :]

            if use_nucleus:
                idx = nucleus_sampling(
                    logits,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    generated_tokens=generated_token_ids,
                )
            else:
                idx = logits.argmax().item()

            if idx == EOS_token:
                break

            generated_token_ids.append(idx)

            if idx in output_lang.index2word:
                decoded_words.append(output_lang.index2word[idx])

            decoder_input = torch.tensor([[idx]], device=device, dtype=torch.long)

    return decoded_words


def load_from_hub(
    model_type="attention", repo_id="tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course"
):
    """
    Download model checkpoint from Hugging Face Hub.

    Args:
        model_type: 'no_attention' or 'attention'
        repo_id: Hugging Face Hub repository ID

    Returns:
        path to downloaded checkpoint
    """
    if not HAS_HF_HUB:
        raise ImportError(
            "huggingface_hub required for --from_hub. Install: pip install huggingface_hub"
        )

    print(f"Downloading {model_type} model from Hugging Face Hub...")

    model_path = hf_hub_download(
        repo_id=repo_id, filename=f"{model_type}/best_model.pth", repo_type="model"
    )

    print(f"✓ Model downloaded from Hub: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Translate using trained model")
    parser.add_argument("--data_source", type=str, default="local")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, default="attention", choices=["no_attention", "attention"]
    )
    parser.add_argument("--from_hub", action="store_true", help="Load model from Hugging Face Hub")
    parser.add_argument(
        "--repo_id", type=str, default="tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of layers (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--use_residual",
        action="store_true",
        default=True,
        help="Use residual connections (must match training)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sentence", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument(
        "--use_nucleus",
        action="store_true",
        default=True,
        help="Use nucleus sampling for generation (default: True)",
    )
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value for nucleus sampling")
    parser.add_argument("--top_p", type=float, default=0.6, help="Top-p value for nucleus sampling")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower=more deterministic, better for translation)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (>1.0 = penalize)",
    )

    args = parser.parse_args()

    # Determine model path
    if args.from_hub:
        args.model_path = load_from_hub(model_type=args.model_type, repo_id=args.repo_id)
    elif args.model_path is None:
        # Default local path
        args.model_path = f"checkpoints_{args.model_type}/best_model.pth"

    device = get_device()

    # Load checkpoint first to get vocabulary and model config
    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Try to load vocabulary from checkpoint (if available)
    if "input_lang" in checkpoint and "output_lang" in checkpoint:
        print("✓ Loading vocabulary from checkpoint")
        input_lang = checkpoint["input_lang"]
        output_lang = checkpoint["output_lang"]
        # Still need to load data for test_loader
        _, _, _, _, test_loader = prepareData(
            data_source=args.data_source,
            lang1="ger",
            lang2="eng",
            reverse=False,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        print("⚠ Vocabulary not in checkpoint, creating from data...")
        from utils import Lang

        # Get vocab sizes from checkpoint weights
        input_vocab_size = checkpoint["encoder_state_dict"]["embedding.weight"].shape[0]
        output_vocab_size = checkpoint["decoder_state_dict"]["embedding.weight"].shape[0]
        print(f"Checkpoint vocab sizes - Input: {input_vocab_size}, Output: {output_vocab_size}")

        # Load data to get vocabulary
        temp_input_lang, temp_output_lang, _, _, test_loader = prepareData(
            data_source=args.data_source,
            lang1="ger",
            lang2="eng",
            reverse=False,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

        # If vocab sizes match, use them
        if (
            temp_input_lang.n_words == input_vocab_size
            and temp_output_lang.n_words == output_vocab_size
        ):
            print("✓ Vocabulary from data matches checkpoint!")
            input_lang = temp_input_lang
            output_lang = temp_output_lang
        else:
            # Vocab mismatch - force vocab size to match checkpoint exactly
            print(
                f"⚠ Forcing vocabulary size to match checkpoint (data: {temp_input_lang.n_words}/{temp_output_lang.n_words} -> checkpoint: {input_vocab_size}/{output_vocab_size})"
            )
            input_lang = temp_input_lang
            output_lang = temp_output_lang

            # Force the n_words to match checkpoint (model will be created with this size)
            input_lang.n_words = input_vocab_size
            output_lang.n_words = output_vocab_size

            print(
                "ℹ️  Note: Vocabulary adjusted to match checkpoint. Inference should work but results may vary."
            )

    print(f"Final vocabulary sizes - Input: {input_lang.n_words}, Output: {output_lang.n_words}")

    # Auto-detect hidden_size and num_layers from checkpoint if not specified
    encoder_embedding_shape = checkpoint["encoder_state_dict"]["embedding.weight"].shape
    if args.hidden_size is None:  # Auto-detect if not specified
        args.hidden_size = encoder_embedding_shape[1]
        print(f"✓ Auto-detected hidden_size: {args.hidden_size}")
    if args.num_layers is None:  # Auto-detect if not specified
        args.num_layers = len(
            [
                k
                for k in checkpoint["encoder_state_dict"].keys()
                if "gru_layers" in k and "weight_ih_l0" in k
            ]
        )
        print(f"✓ Auto-detected num_layers: {args.num_layers}")

    # Create models (choose between standard or residual)
    # Use dropout=0.1 to match training configuration (from config files)
    dropout_p = 0.1

    if args.use_residual:
        print("Using Residual models with Layer Normalization")
        encoder = ResidualStackedGRUEncoder(
            input_lang.n_words, args.hidden_size, num_layers=args.num_layers, dropout_p=dropout_p
        ).to(device)

        if args.model_type == "attention":
            decoder = ResidualStackedGRUAttnDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=dropout_p,
                max_length=args.max_length,
            ).to(device)
        else:
            decoder = ResidualStackedGRUDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=dropout_p,
            ).to(device)
    else:
        print("Using Standard stacked models")
        encoder = StackedGRUEncoder(
            input_lang.n_words, args.hidden_size, num_layers=args.num_layers, dropout_p=dropout_p
        ).to(device)

        if args.model_type == "attention":
            decoder = StackedGRUAttnDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=dropout_p,
                max_length=args.max_length,
            ).to(device)
        else:
            decoder = StackedGRUDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=dropout_p,
            ).to(device)

    # Load model weights (checkpoint already loaded above)
    print("Loading model weights...")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    print("✓ Model loaded successfully")

    encoder.eval()
    decoder.eval()

    if args.sentence:
        output_words = translate_sentence(
            encoder,
            decoder,
            args.sentence,
            input_lang,
            output_lang,
            device,
            args.max_length,
            use_nucleus=args.use_nucleus,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        output_sentence = " ".join(output_words)
        print(f"Input: {args.sentence}")
        print(f"Translation: {output_sentence}")
    else:

        import random

        all_pairs = []
        for input_tensor, target_tensor in test_loader:
            for i in range(input_tensor.size(0)):
                src_words = []
                for idx in input_tensor[i].tolist():
                    if idx == EOS_token:
                        break
                    if idx in input_lang.index2word:
                        src_words.append(input_lang.index2word[idx])

                tgt_words = []
                for idx in target_tensor[i].tolist():
                    if idx == EOS_token:
                        break
                    if idx in output_lang.index2word:
                        tgt_words.append(output_lang.index2word[idx])

                all_pairs.append((" ".join(src_words), " ".join(tgt_words)))

        sample_pairs = random.sample(all_pairs, min(args.num_examples, len(all_pairs)))

        for i, (src, tgt) in enumerate(sample_pairs, 1):
            output_words = translate_sentence(
                encoder,
                decoder,
                src,
                input_lang,
                output_lang,
                device,
                args.max_length,
                use_nucleus=args.use_nucleus,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
            output_sentence = " ".join(output_words)
            print(f"\nExample {i}:")
            print(f"  Input:  {src}")
            print(f"  Target: {tgt}")
            print(f"  Output: {output_sentence}")


if __name__ == "__main__":
    main()
