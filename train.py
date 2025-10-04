"""
Training script with distributed training and modular architecture.
"""

import os
import time
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.cuda.amp import GradScaler

import bitsandbytes as bnb
from tqdm import tqdm
import wandb

# Import core modules
from data_loader import prepareData
from model import (
    StackedGRUEncoder,
    StackedGRUDecoder,
    StackedGRUAttnDecoder,
    ResidualStackedGRUEncoder,
    ResidualStackedGRUDecoder,
    ResidualStackedGRUAttnDecoder,
)

# Import from unified utils module
from utils import (
    get_device,
    calculate_perplexity,
    SOS_token,
    EOS_token,
    generate_report,
    save_generations,
    setup_distributed,
    cleanup_distributed,
    print_model_info,
)
from evaluation import (
    evaluate_val_loss_teacher_forcing,
    sample_and_translate,
    evaluate_metrics,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_step(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_grad_norm,
    device,
    scaler=None,
    use_amp=True,
    amp_dtype=torch.bfloat16,
):
    """Single training step with mixed precision, gradient clipping and teacher forcing."""
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_tensor.size(0)

    # Use autocast for mixed precision
    with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        encoder_hidden = (
            encoder.module.initHidden(batch_size, device)
            if hasattr(encoder, "module")
            else encoder.initHidden(batch_size, device)
        )
        # Input tensors remain as Long for embedding layers
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        decoder_hidden = encoder_hidden

        # Teacher forcing: use ground truth tokens as decoder input
        decoder_input = torch.cat(
            [
                torch.full((target_tensor.size(0), 1), SOS_token, device=device, dtype=torch.long),
                target_tensor[:, :-1],
            ],
            dim=1,
        )

        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

        loss = criterion(
            decoder_output.reshape(-1, decoder_output.size(-1)), target_tensor.reshape(-1)
        )

    # Backward pass with or without scaler (bfloat16 doesn't need scaler)
    if scaler is not None and amp_dtype == torch.float16:
        scaler.scale(loss).backward()
        scaler.unscale_(encoder_optimizer)
        scaler.unscale_(decoder_optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)
        scaler.step(encoder_optimizer)
        scaler.step(decoder_optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item()


def train(
    encoder,
    decoder,
    train_loader,
    val_loader,
    input_lang,
    output_lang,
    device,
    rank,
    n_epochs=10,
    learning_rate=0.001,
    print_every=100,
    eval_every=1000,
    save_dir="checkpoints",
    max_length=64,
    max_grad_norm=1.0,
    use_wandb=False,
    use_amp=True,
    amp_dtype="bfloat16",
):
    """Main training loop with mixed precision, validation, checkpointing, wandb, and tqdm."""
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    # Track validation metrics for plotting
    val_losses = []
    val_perplexities = []
    val_steps = []

    # Convert amp_dtype string to torch dtype
    if amp_dtype == "float16":
        amp_dtype_torch = torch.float16
    elif amp_dtype == "bfloat16":
        amp_dtype_torch = torch.bfloat16
    else:
        amp_dtype_torch = torch.float32
        use_amp = False

    # Initialize GradScaler only for float16 (not needed for bfloat16)
    scaler = GradScaler() if (use_amp and amp_dtype_torch == torch.float16) else None

    if rank == 0:
        logger.info(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'} (dtype: {amp_dtype})")
        if scaler:
            logger.info("Using GradScaler for FP16 training")

    # Calculate eval_every to validate 2 times per epoch (every 1/2 epoch)
    steps_per_epoch = len(train_loader)
    eval_every = max(1, steps_per_epoch // 2)

    if rank == 0:
        logger.info(
            f"Validating every {eval_every} steps (2 times per epoch of {steps_per_epoch} steps)"
        )

    encoder_optimizer = bnb.optim.AdamW8bit(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = bnb.optim.AdamW8bit(decoder.parameters(), lr=learning_rate)

    total_steps = n_epochs * len(train_loader)
    encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        encoder_optimizer, T_max=total_steps
    )
    decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        decoder_optimizer, T_max=total_steps
    )

    # No GradScaler needed for bfloat16 - it has better numerical stability
    criterion = nn.NLLLoss(ignore_index=0)

    global_step = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        num_batches = 0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        else:
            pbar = train_loader

        for batch_idx, (input_tensor, target_tensor) in enumerate(pbar):
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss = train_step(
                input_tensor,
                target_tensor,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
                max_grad_norm,
                device,
                scaler=scaler,
                use_amp=use_amp,
                amp_dtype=amp_dtype_torch,
            )

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            encoder_scheduler.step()
            decoder_scheduler.step()

            if rank == 0:
                avg_loss = epoch_loss / num_batches
                perplexity = calculate_perplexity(avg_loss)

                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ppl": f"{perplexity:.2f}"})

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/perplexity": perplexity,
                            "train/lr": encoder_scheduler.get_last_lr()[0],
                            "train/step": global_step,
                        }
                    )

            if rank == 0 and global_step % eval_every == 0:
                val_loss = evaluate_val_loss_teacher_forcing(
                    encoder,
                    decoder,
                    val_loader,
                    device,
                    criterion,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype_torch,
                )
                val_perplexity = calculate_perplexity(val_loss)

                # Sample 2 validation instances for translation tracking
                sample_translations = sample_and_translate(
                    encoder,
                    decoder,
                    val_loader,
                    input_lang,
                    output_lang,
                    device,
                    max_length=max_length,
                    num_samples=2,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype_torch,
                )

                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": val_perplexity,
                            "val/step": global_step,
                        }
                    )

                    # Log sample translations to track improvement
                    if sample_translations:
                        translation_table = wandb.Table(
                            columns=["Step", "Source", "Reference", "Prediction"],
                            data=[
                                [global_step, src, ref, pred]
                                for src, ref, pred in sample_translations
                            ],
                        )
                        wandb.log(
                            {"val/sample_translations": translation_table, "val/step": global_step}
                        )

                # Track validation metrics for plotting
                val_losses.append(val_loss)
                val_perplexities.append(val_perplexity)
                val_steps.append(global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    model_to_save_encoder = (
                        encoder.module if hasattr(encoder, "module") else encoder
                    )
                    model_to_save_decoder = (
                        decoder.module if hasattr(decoder, "module") else decoder
                    )

                    torch.save(
                        {
                            "encoder_state_dict": model_to_save_encoder.state_dict(),
                            "decoder_state_dict": model_to_save_decoder.state_dict(),
                            "val_loss": val_loss,
                            "epoch": epoch,
                            "global_step": global_step,
                            "input_lang": input_lang,
                            "output_lang": output_lang,
                        },
                        os.path.join(save_dir, "best_model.pth"),
                    )

                    if use_wandb:
                        wandb.run.summary["best_val_loss"] = best_val_loss
                        wandb.run.summary["best_val_perplexity"] = val_perplexity

        if rank == 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_perplexity = calculate_perplexity(avg_epoch_loss)

    # Save validation loss/perplexity curves after training (rank 0 only)
    if rank == 0 and len(val_losses) > 0:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot validation loss
        ax1.plot(val_steps, val_losses, "b-", linewidth=2, marker="o", markersize=6)
        ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_title("Validation Loss over Training", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot validation perplexity
        ax2.plot(val_steps, val_perplexities, "r-", linewidth=2, marker="s", markersize=6)
        ax2.set_xlabel("Training Step", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Validation Perplexity", fontsize=12, fontweight="bold")
        ax2.set_title("Validation Perplexity over Training", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "validation_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Validation curves saved to {plot_path}")

        # Save raw validation data for comparison chart
        with open(os.path.join(save_dir, "val_losses.txt"), "w") as f:
            for loss in val_losses:
                f.write(f"{loss}\n")
        with open(os.path.join(save_dir, "val_perplexities.txt"), "w") as f:
            for ppl in val_perplexities:
                f.write(f"{ppl}\n")

        # Log to wandb if enabled
        if use_wandb:
            wandb.log({"val/curves": wandb.Image(plot_path)})


def main():
    parser = argparse.ArgumentParser(description="Train Seq2seq Model with DDP")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument(
        "--data_source", type=str, default="local", help="Data source: local or tatoeba"
    )
    parser.add_argument(
        "--model_type", type=str, default="attention", choices=["no_attention", "attention"]
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument(
        "--wandb_project", type=str, default="translation-de-en", help="W&B project name"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--use_amp", action="store_true", default=True, help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="AMP dtype",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        default=False,
        help="Use residual connections with layer normalization",
    )

    args = parser.parse_args()

    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0 and args.use_wandb:
        run_name = (
            args.wandb_run_name or f"{args.model_type}_h{args.hidden_size}_bs{args.batch_size}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    input_lang, output_lang, train_loader, val_loader, test_loader = prepareData(
        data_source=args.data_source,
        lang1="ger",
        lang2="eng",
        reverse=False,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create models (keep in float32, autocast will handle mixed precision)
    # Choose between standard or residual models
    if args.use_residual:
        if rank == 0:
            logger.info("Using Residual models with Layer Normalization")
        encoder = ResidualStackedGRUEncoder(
            input_lang.n_words, args.hidden_size, num_layers=args.num_layers, dropout_p=args.dropout
        ).to(device)

        if args.model_type == "attention":
            decoder = ResidualStackedGRUAttnDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=args.dropout,
                max_length=args.max_length,
            ).to(device)
        else:
            decoder = ResidualStackedGRUDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=args.dropout,
            ).to(device)
    else:
        if rank == 0:
            logger.info("Using Standard stacked models")
        encoder = StackedGRUEncoder(
            input_lang.n_words, args.hidden_size, num_layers=args.num_layers, dropout_p=args.dropout
        ).to(device)

        if args.model_type == "attention":
            decoder = StackedGRUAttnDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=args.dropout,
                max_length=args.max_length,
            ).to(device)
        else:
            decoder = StackedGRUDecoder(
                args.hidden_size,
                output_lang.n_words,
                num_layers=args.num_layers,
                dropout_p=args.dropout,
            ).to(device)

    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
        decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)

    # Print model information before training
    print_model_info(encoder, decoder, rank)

    train(
        encoder,
        decoder,
        train_loader,
        val_loader,
        input_lang,
        output_lang,
        device,
        rank,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        use_wandb=args.use_wandb and rank == 0,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
    )

    # Synchronize all processes before final evaluation
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info("Starting final evaluation on test set...")

        checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pth"), weights_only=False)

        model_encoder = encoder.module if hasattr(encoder, "module") else encoder
        model_decoder = decoder.module if hasattr(decoder, "module") else decoder

        model_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        criterion = nn.NLLLoss(ignore_index=0)

        # Get amp settings
        amp_dtype_torch = (
            torch.bfloat16
            if args.amp_dtype == "bfloat16"
            else (torch.float16 if args.amp_dtype == "float16" else torch.float32)
        )

        test_loss = evaluate_val_loss_teacher_forcing(
            model_encoder,
            model_decoder,
            test_loader,
            device,
            criterion,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype_torch,
        )
        (
            test_accuracy,
            test_bleu,
            test_bertscore,
            test_bleurt,
            test_comet,
            predictions,
            references,
            sources,
        ) = evaluate_metrics(
            model_encoder,
            model_decoder,
            test_loader,
            input_lang,
            output_lang,
            device,
            args.max_length,
            use_amp=args.use_amp,
            amp_dtype=amp_dtype_torch,
        )

        # Ensure analysis_results directory exists
        os.makedirs("analysis_results", exist_ok=True)
        report_name = f"analysis_results/comparison_report_{args.model_type}.txt"
        generate_report(
            args.model_type,
            test_accuracy,
            test_bleu,
            test_bertscore,
            test_bleurt,
            test_comet,
            test_loss,
            predictions,
            references,
            sources,
            report_name,
        )

        # Save generation results to file
        generations_file = os.path.join(args.save_dir, f"test_generations_{args.model_type}.txt")
        save_generations(predictions, references, generations_file)
        logger.info(f"Test generations saved to {generations_file}")

        if args.use_wandb:
            # Log test metrics
            wandb.log(
                {
                    "test/loss": test_loss,
                    "test/perplexity": calculate_perplexity(test_loss),
                    "test/accuracy": test_accuracy,
                    "test/bleu": test_bleu,
                }
            )

            # Create comparison table for wandb
            test_results_table = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Test Loss", f"{test_loss:.4f}"],
                    ["Test Perplexity", f"{calculate_perplexity(test_loss):.4f}"],
                    ["Test Accuracy", f"{test_accuracy:.4f}"],
                    ["Test BLEU", f"{test_bleu:.4f}"],
                    ["Samples Evaluated", str(len(predictions))],
                ],
            )
            wandb.log({"test/metrics_table": test_results_table})

            # Log sample translations
            translation_table = wandb.Table(
                columns=["Sample", "Reference", "Prediction"],
                data=[
                    [i + 1, ref, pred]
                    for i, (ref, pred) in enumerate(zip(references[:20], predictions[:20]))
                ],
            )
            wandb.log({"test/sample_translations": translation_table})

            # Log generation file as artifact
            artifact = wandb.Artifact(f"test_generations_{args.model_type}", type="predictions")
            artifact.add_file(generations_file)
            wandb.log_artifact(artifact)

            wandb.finish()

        logger.info("Final evaluation completed successfully!")

    # Synchronize all processes before cleanup
    if world_size > 1:
        dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
