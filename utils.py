"""
Unified utility functions and classes for translation project.

Includes:
- Language vocabulary management
- Text processing and normalization
- Device detection (MPS/CUDA/CPU)
- Metrics calculation (BLEU, BERTScore, BLEURT, COMET)
- Sampling strategies (nucleus sampling)
- Model utilities (parameter counting, size calculation)
- Distributed training utilities (DDP setup/cleanup)
- Plotting and reporting
"""

import os
import re
import math
import time
import logging
import unicodedata
import torch
import torch.distributed as dist
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

logger = logging.getLogger(__name__)

SOS_token = 0
EOS_token = 1


# ============================================================================
# DISTRIBUTED TRAINING UTILITIES
# ============================================================================


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# MODEL UTILITIES
# ============================================================================


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model, dtype=torch.float16):
    """Calculate model size in MB for a given dtype."""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * dtype.itemsize
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(encoder, decoder, rank=0):
    """Print model parameter count and size information."""
    if rank != 0:
        return

    encoder_model = encoder.module if hasattr(encoder, "module") else encoder
    decoder_model = decoder.module if hasattr(decoder, "module") else decoder

    encoder_params = count_parameters(encoder_model)
    decoder_params = count_parameters(decoder_model)
    total_params = encoder_params + decoder_params

    encoder_size_fp16 = get_model_size_mb(encoder_model, torch.float16)
    decoder_size_fp16 = get_model_size_mb(decoder_model, torch.float16)
    total_size_fp16 = encoder_size_fp16 + decoder_size_fp16

    encoder_size_bf16 = get_model_size_mb(encoder_model, torch.bfloat16)
    decoder_size_bf16 = get_model_size_mb(decoder_model, torch.bfloat16)
    total_size_bf16 = encoder_size_bf16 + decoder_size_bf16

    logger.info("=" * 80)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Encoder Parameters: {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    logger.info(f"Decoder Parameters: {decoder_params:,} ({decoder_params/1e6:.2f}M)")
    logger.info(f"Total Parameters:   {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info("-" * 80)
    logger.info(f"Encoder Size (FP16):  {encoder_size_fp16:.2f} MB")
    logger.info(f"Decoder Size (FP16):  {decoder_size_fp16:.2f} MB")
    logger.info(f"Total Size (FP16):    {total_size_fp16:.2f} MB")
    logger.info("-" * 80)
    logger.info(f"Encoder Size (BF16):  {encoder_size_bf16:.2f} MB")
    logger.info(f"Decoder Size (BF16):  {decoder_size_bf16:.2f} MB")
    logger.info(f"Total Size (BF16):    {total_size_bf16:.2f} MB")
    logger.info("=" * 80)


# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================


def nucleus_sampling(
    logits, top_k=20, top_p=0.6, temperature=0.7, repetition_penalty=1.05, generated_tokens=None
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
        sampled token index (int)
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


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def calculate_bertscore(predictions, references, device="cuda"):
    """Calculate BERTScore using bert_score library directly (fixed for compatibility)."""
    try:
        # Use bert_score directly instead of evaluate library
        from bert_score import score as bert_score_func

        P, R, F1 = bert_score_func(
            predictions,
            references,
            model_type="distilbert-base-uncased",
            device=device,
            batch_size=16,
            verbose=False,
        )
        # Return average F1 score
        return F1.mean().item()
    except Exception as e:
        logger.warning(f"BERTScore calculation failed: {e}, returning 0.0")
        return 0.0


def calculate_bleurt(predictions, references):
    """
    Calculate BLEURT using evaluate library (with error handling).

    Note: BLEURT is optional and requires: pip install git+https://github.com/google-research/bleurt.git
    If not installed, returns 0.0 (other metrics like BERTScore and COMET still work).
    """
    try:
        # Try using evaluate library with error handling
        import evaluate

        bleurt = evaluate.load("bleurt", "BLEURT-20-D3", trust_remote_code=True)
        results = bleurt.compute(predictions=predictions, references=references)
        return sum(results["scores"]) / len(results["scores"])
    except Exception as e:
        logger.info(
            "BLEURT not available (optional metric). "
            "Install with: pip install git+https://github.com/google-research/bleurt.git"
        )
        logger.debug(f"BLEURT error details: {e}")
        return 0.0


def calculate_comet(predictions, references, sources):
    """Calculate COMET using comet library directly (fixed for compatibility)."""
    try:
        # Use comet library directly
        from comet import download_model, load_from_checkpoint

        # Download and load the model
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)

        # Prepare data format
        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        # Get predictions
        model_output = model.predict(
            data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0
        )
        return model_output["system_score"]
    except Exception as e:
        logger.warning(f"COMET calculation failed: {e}, returning 0.0")
        return 0.0


# ============================================================================
# LANGUAGE & TEXT PROCESSING
# ============================================================================


class Lang:
    """Language vocabulary class for managing word-to-index mappings."""

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalizeString(s):
    """Lowercase, trim, and remove non-letter characters."""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p, max_length=64):
    """Filter pairs based on maximum length."""
    return len(p[0].split(" ")) < max_length and len(p[1].split(" ")) < max_length


def indexesFromSentence(lang, sentence):
    """Convert sentence to list of word indexes."""
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence, device):
    """Convert sentence to tensor of word indexes."""
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang, device):
    """Convert a pair of sentences to tensors."""
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


def get_device():
    """Get the available device (MPS, CUDA, or CPU) with auto-detection."""
    if torch.backends.mps.is_available():
        # Apple Silicon GPU (M1/M2/M3)
        return torch.device("mps")
    elif torch.cuda.is_available():
        # NVIDIA GPU
        return torch.device("cuda")
    else:
        # CPU fallback
        return torch.device("cpu")


def asMinutes(s):
    """Convert seconds to minutes and seconds."""
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    """Calculate elapsed and remaining time."""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def showPlot(points, title="Training Loss", save_path=None):
    """Plot training curves."""
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    return math.exp(loss)


def calculate_bleu(predictions, references):
    """
    Calculate BLEU score using NLTK.

    Args:
        predictions: List of predicted sentences (strings)
        references: List of reference sentences (strings)

    Returns:
        BLEU score (float)
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    smoothie = SmoothingFunction().method4

    # Convert to list of tokens for NLTK
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]

    # Calculate corpus BLEU score
    bleu_score = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
    return bleu_score


def sentence_accuracy(pred_tokens, ref_tokens):
    """Calculate token-level accuracy for a sentence pair."""
    min_len = min(len(pred_tokens), len(ref_tokens))
    correct = sum(1 for i in range(min_len) if pred_tokens[i] == ref_tokens[i])
    return correct / max(len(ref_tokens), 1)


def save_generations(predictions, references, save_path):
    """Save generation results to file."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*120}\n")
        f.write(f"TEST SET GENERATION RESULTS\n")
        f.write(f"{'='*120}\n\n")
        f.write(f"Total samples: {len(predictions)}\n\n")

        for i, (pred, ref) in enumerate(zip(predictions, references), 1):
            f.write(f"Sample {i}:\n")
            f.write(f"Reference  : {ref}\n")
            f.write(f"Prediction : {pred}\n")
            f.write(f"{'-'*120}\n")


def generate_report(
    model_name,
    test_accuracy,
    test_bleu,
    test_bertscore,
    test_bleurt,
    test_comet,
    test_loss,
    predictions,
    references,
    sources,
    save_path="comparison_report.txt",
):
    """Generate evaluation report with multiple metrics."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*120}\n")
        f.write(f"EVALUATION REPORT: {model_name}\n")
        f.write(f"{'='*120}\n\n")
        f.write(f"Quantitative Results (evaluated on {len(predictions)} samples):\n")
        f.write(f"{'-'*120}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Perplexity: {calculate_perplexity(test_loss):.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test BLEU Score: {test_bleu:.4f}\n")
        f.write(f"Test BERTScore (F1): {test_bertscore:.4f}\n")
        f.write(f"Test BLEURT: {test_bleurt:.4f}\n")
        f.write(f"Test COMET: {test_comet:.4f}\n")
        f.write(f"Number of samples evaluated: {len(predictions)}\n\n")
        f.write(f"Qualitative Results (Sample Translations):\n")
        f.write(f"{'-'*120}\n")
        sample_indices = random.sample(range(len(predictions)), min(20, len(predictions)))
        for i, idx in enumerate(sample_indices, 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"Source: {sources[idx]}\n")
            f.write(f"Reference: {references[idx]}\n")
            f.write(f"Predicted: {predictions[idx]}\n")
        f.write(f"\n{'='*120}\n")
