"""
Evaluation functions for translation models.
"""

import torch
import logging
from torch.amp import autocast
from nltk.translate.bleu_score import SmoothingFunction

from utils import (
    calculate_bleu,
    sentence_accuracy,
    SOS_token,
    EOS_token,
    nucleus_sampling,
    calculate_bertscore,
    calculate_bleurt,
    calculate_comet,
)

logger = logging.getLogger(__name__)


def evaluate_val_loss_teacher_forcing(
    encoder, decoder, val_loader, device, criterion, use_amp=True, amp_dtype=torch.bfloat16
):
    """Evaluate validation loss using teacher forcing with mixed precision."""
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for input_tensor, target_tensor in val_loader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            batch_size = input_tensor.size(0)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                encoder_hidden = (
                    encoder.module.initHidden(batch_size, device)
                    if hasattr(encoder, "module")
                    else encoder.initHidden(batch_size, device)
                )
                # Input tensors remain as Long for embedding layers
                encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

                decoder_hidden = encoder_hidden
                # Teacher forcing for validation loss calculation
                decoder_input = torch.cat(
                    [
                        torch.full(
                            (target_tensor.size(0), 1), SOS_token, device=device, dtype=torch.long
                        ),
                        target_tensor[:, :-1],
                    ],
                    dim=1,
                )

                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                loss = criterion(
                    decoder_output.reshape(-1, decoder_output.size(-1)), target_tensor.reshape(-1)
                )

            total_loss += loss.item()
            total_batches += 1

    encoder.train()
    decoder.train()

    return total_loss / total_batches


def translate_batch(encoder, decoder, input_tensor, output_lang, device, max_length=64):
    """Translate a batch of sentences."""
    encoder.eval()
    decoder.eval()

    translations = []
    batch_size = input_tensor.size(0)

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            encoder_hidden = (
                encoder.module.initHidden(device)
                if hasattr(encoder, "module")
                else encoder.initHidden(device)
            )
            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

            decoder_input = torch.full((batch_size, 1), SOS_token, device=device)
            decoder_hidden = encoder_hidden

            for _ in range(max_length):
                decoder_output, decoder_hidden, _ = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                topv, topi = decoder_output[:, -1, :].topk(1)
                decoder_input = topi

                if (topi == EOS_token).all():
                    break

    encoder.train()
    decoder.train()

    return translations


def sample_and_translate(
    encoder,
    decoder,
    val_loader,
    input_lang,
    output_lang,
    device,
    max_length=64,
    num_samples=2,
    use_amp=True,
    amp_dtype=torch.bfloat16,
):
    """Sample a few instances from validation set and translate them with greedy decoding."""
    encoder.eval()
    decoder.eval()

    samples = []

    with torch.no_grad():
        for input_tensor, target_tensor in val_loader:
            if len(samples) >= num_samples:
                break

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            for i in range(min(num_samples - len(samples), input_tensor.size(0))):
                single_input = input_tensor[i : i + 1]
                single_target = target_tensor[i]

                # Get source sentence
                source_words = []
                for idx in single_input[0].cpu().tolist():
                    if idx == EOS_token:
                        break
                    if idx in input_lang.index2word:
                        source_words.append(input_lang.index2word[idx])
                source_sentence = " ".join(source_words)

                # Get reference translation
                target_words = []
                for idx in single_target.cpu().tolist():
                    if idx == EOS_token:
                        break
                    if idx in output_lang.index2word:
                        target_words.append(output_lang.index2word[idx])
                reference_sentence = " ".join(target_words)

                # Greedy decoding with mixed precision
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    encoder_hidden = (
                        encoder.module.initHidden(1, device)
                        if hasattr(encoder, "module")
                        else encoder.initHidden(1, device)
                    )
                    encoder_outputs, encoder_hidden = encoder(single_input, encoder_hidden)

                    decoder_input = torch.full((1, 1), SOS_token, device=device, dtype=torch.long)
                    decoder_hidden = encoder_hidden

                    decoded_words = []
                    for _ in range(max_length):
                        decoder_output, decoder_hidden, _ = decoder(
                            decoder_input, decoder_hidden, encoder_outputs
                        )

                        # Greedy decoding (argmax)
                        topv, topi = decoder_output[:, -1, :].topk(1)
                        idx = topi.item()

                        if idx == EOS_token:
                            break

                        if idx in output_lang.index2word:
                            decoded_words.append(output_lang.index2word[idx])

                        decoder_input = topi

                prediction_sentence = " ".join(decoded_words)
                samples.append((source_sentence, reference_sentence, prediction_sentence))

                if len(samples) >= num_samples:
                    break

    encoder.train()
    decoder.train()

    return samples


def evaluate_metrics(
    encoder,
    decoder,
    test_loader,
    input_lang,
    output_lang,
    device,
    max_length=64,
    max_samples=512,
    use_amp=True,
    amp_dtype=torch.bfloat16,
):
    """
    Evaluate model with BLEU and accuracy metrics using auto-regressive generation with nucleus sampling.

    Limited to 512 samples for fast computation. Uses nucleus sampling with:
    - top_k=20, top_p=0.6, temperature=0.7, repetition_penalty=1.05
    """
    encoder.eval()
    decoder.eval()

    predictions = []
    references = []
    sources = []
    accuracies = []

    smoothie = SmoothingFunction().method4
    total_samples = 0

    with torch.no_grad():
        for input_tensor, target_tensor in test_loader:
            if total_samples >= max_samples:
                break
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            batch_size = input_tensor.size(0)

            for i in range(batch_size):
                if total_samples >= max_samples:
                    break

                single_input = input_tensor[i : i + 1]
                single_target = target_tensor[i]

                # Get source sentence (German)
                source_words = []
                for idx in single_input[0].cpu().tolist():
                    if idx == EOS_token:
                        break
                    if idx in input_lang.index2word:
                        source_words.append(input_lang.index2word[idx])
                source_sentence = " ".join(source_words)

                with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    encoder_hidden = (
                        encoder.module.initHidden(1, device)
                        if hasattr(encoder, "module")
                        else encoder.initHidden(1, device)
                    )
                    # Input tensors remain as Long for embedding layers
                    encoder_outputs, encoder_hidden = encoder(single_input, encoder_hidden)

                    # Auto-regressive generation with nucleus sampling (no teacher forcing)
                    decoder_input = torch.full((1, 1), SOS_token, device=device, dtype=torch.long)
                    decoder_hidden = encoder_hidden

                    decoded_words = []
                    generated_token_ids = []
                    for _ in range(max_length):
                        decoder_output, decoder_hidden, _ = decoder(
                            decoder_input, decoder_hidden, encoder_outputs
                        )

                        logits = decoder_output[0, -1, :]
                        # Use nucleus sampling: top_k=20, top_p=0.6, temperature=0.7, repetition_penalty=1.05
                        idx = nucleus_sampling(
                            logits,
                            top_k=20,
                            top_p=0.6,
                            temperature=0.7,
                            repetition_penalty=1.05,
                            generated_tokens=generated_token_ids,
                        )

                        if idx == EOS_token:
                            break

                        generated_token_ids.append(idx)

                        if idx in output_lang.index2word:
                            decoded_words.append(output_lang.index2word[idx])

                        decoder_input = torch.tensor([[idx]], device=device, dtype=torch.long)

                output_sentence = " ".join(decoded_words)

                target_words = []
                for idx in single_target.cpu().tolist():
                    if idx == EOS_token:
                        break
                    if idx in output_lang.index2word:
                        target_words.append(output_lang.index2word[idx])

                target_sentence = " ".join(target_words)

                sources.append(source_sentence)
                predictions.append(output_sentence)
                references.append(target_sentence)

                acc = sentence_accuracy(decoded_words, target_words)
                accuracies.append(acc)

                total_samples += 1

    encoder.train()
    decoder.train()

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    bleu_score = calculate_bleu(predictions, references)

    # Calculate additional metrics: BERTScore, BLEURT, COMET
    logger.info("Calculating BERTScore...")
    bertscore_f1 = calculate_bertscore(predictions, references, device=str(device))

    logger.info("Calculating BLEURT...")
    bleurt_score = calculate_bleurt(predictions, references)

    logger.info("Calculating COMET...")
    comet_score = calculate_comet(predictions, references, sources)

    return (
        avg_accuracy,
        bleu_score,
        bertscore_f1,
        bleurt_score,
        comet_score,
        predictions,
        references,
        sources,
    )
