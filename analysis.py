"""
Analysis script for comparing model results.
"""

import argparse
import logging
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_report(report_path):
    """Parse comparison report and extract metrics."""
    metrics = {}

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            if "Test Loss:" in line:
                metrics["loss"] = float(line.split(":")[1].strip())
            elif "Test Perplexity:" in line:
                metrics["perplexity"] = float(line.split(":")[1].strip())
            elif "Test Accuracy:" in line:
                metrics["accuracy"] = float(line.split(":")[1].strip())
            elif "Test BLEU Score:" in line:
                metrics["bleu"] = float(line.split(":")[1].strip())
            elif "Best Val Loss:" in line:
                metrics["val_loss"] = float(line.split(":")[1].strip())
            elif "Best Val Perplexity:" in line:
                metrics["val_perplexity"] = float(line.split(":")[1].strip())

    return metrics


def create_comparison_table(no_attn_metrics, attn_metrics, save_path="comparison_table.txt"):
    """Create formatted comparison table."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON: NO ATTENTION vs WITH ATTENTION\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Metric':<25} {'No Attention':<20} {'With Attention':<20} {'Improvement':<25}\n")
        f.write("-" * 90 + "\n")

        # Test metrics
        f.write("\nTest Set Metrics:\n")
        for metric in ["loss", "perplexity", "accuracy", "bleu"]:
            no_attn_val = no_attn_metrics.get(metric, 0)
            attn_val = attn_metrics.get(metric, 0)

            if metric in ["loss", "perplexity"]:
                if no_attn_val != 0:
                    improvement = ((no_attn_val - attn_val) / abs(no_attn_val)) * 100
                    improvement_str = f"{improvement:+.2f}% (lower better)"
                else:
                    improvement_str = "N/A (baseline is 0)"
            else:
                if no_attn_val != 0:
                    improvement = ((attn_val - no_attn_val) / abs(no_attn_val)) * 100
                    improvement_str = f"{improvement:+.2f}% (higher better)"
                else:
                    improvement_str = "N/A (baseline is 0)"

            f.write(
                f"  {metric.capitalize():<23} {no_attn_val:<20.4f} {attn_val:<20.4f} {improvement_str:<25}\n"
            )

        # Validation metrics
        f.write("\nValidation Set Metrics:\n")
        for metric in ["val_loss", "val_perplexity"]:
            no_attn_val = no_attn_metrics.get(metric, 0)
            attn_val = attn_metrics.get(metric, 0)

            if no_attn_val != 0:
                improvement = ((no_attn_val - attn_val) / abs(no_attn_val)) * 100
                improvement_str = f"{improvement:+.2f}% (lower better)"
            else:
                improvement_str = "N/A (baseline is 0)"

            metric_display = metric.replace("val_", "").capitalize()
            f.write(
                f"  {metric_display:<23} {no_attn_val:<20.4f} {attn_val:<20.4f} {improvement_str:<25}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("\nSummary:\n")
        f.write("-" * 80 + "\n")

        if attn_metrics["bleu"] > no_attn_metrics["bleu"]:
            f.write("✓ Attention mechanism improves BLEU score\n")
        else:
            f.write("✗ Attention mechanism does not improve BLEU score\n")

        if attn_metrics["accuracy"] > no_attn_metrics["accuracy"]:
            f.write("✓ Attention mechanism improves accuracy\n")
        else:
            f.write("✗ Attention mechanism does not improve accuracy\n")

        if attn_metrics["loss"] < no_attn_metrics["loss"]:
            f.write("✓ Attention mechanism reduces loss\n")
        else:
            f.write("✗ Attention mechanism does not reduce loss\n")

        f.write("\n" + "=" * 80 + "\n")


def create_validation_comparison_chart(
    checkpoint_no_attn, checkpoint_attn, save_path="validation_comparison.png"
):
    """Create combined validation curves comparison chart."""
    import matplotlib.pyplot as plt

    # Read validation data for both models
    val_losses_no_attn = []
    val_perplexities_no_attn = []
    val_losses_attn = []
    val_perplexities_attn = []

    # Read no attention validation data
    loss_file_no_attn = os.path.join(checkpoint_no_attn, "val_losses.txt")
    ppl_file_no_attn = os.path.join(checkpoint_no_attn, "val_perplexities.txt")

    if os.path.exists(loss_file_no_attn) and os.path.exists(ppl_file_no_attn):
        with open(loss_file_no_attn, "r") as f:
            val_losses_no_attn = [float(line.strip()) for line in f if line.strip()]
        with open(ppl_file_no_attn, "r") as f:
            val_perplexities_no_attn = [float(line.strip()) for line in f if line.strip()]

    # Read attention validation data
    loss_file_attn = os.path.join(checkpoint_attn, "val_losses.txt")
    ppl_file_attn = os.path.join(checkpoint_attn, "val_perplexities.txt")

    if os.path.exists(loss_file_attn) and os.path.exists(ppl_file_attn):
        with open(loss_file_attn, "r") as f:
            val_losses_attn = [float(line.strip()) for line in f if line.strip()]
        with open(ppl_file_attn, "r") as f:
            val_perplexities_attn = [float(line.strip()) for line in f if line.strip()]

    if not val_losses_no_attn or not val_losses_attn:
        logger.warning("Validation data not found, skipping validation comparison chart")
        return

    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot validation loss
    steps_no_attn = list(range(len(val_losses_no_attn)))
    steps_attn = list(range(len(val_losses_attn)))

    ax1.plot(
        steps_no_attn,
        val_losses_no_attn,
        "b-o",
        linewidth=2,
        markersize=6,
        label="No Attention",
        alpha=0.8,
    )
    ax1.plot(
        steps_attn,
        val_losses_attn,
        "r-s",
        linewidth=2,
        markersize=6,
        label="With Attention",
        alpha=0.8,
    )
    ax1.set_xlabel("Validation Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot validation perplexity
    ax2.plot(
        steps_no_attn,
        val_perplexities_no_attn,
        "b-o",
        linewidth=2,
        markersize=6,
        label="No Attention",
        alpha=0.8,
    )
    ax2.plot(
        steps_attn,
        val_perplexities_attn,
        "r-s",
        linewidth=2,
        markersize=6,
        label="With Attention",
        alpha=0.8,
    )
    ax2.set_xlabel("Validation Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation Perplexity", fontsize=12, fontweight="bold")
    ax2.set_title("Validation Perplexity Comparison", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Validation comparison chart saved to {save_path}")


def create_comparison_plots(no_attn_metrics, attn_metrics, save_dir="plots"):
    """Create comparison plots."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["loss", "perplexity", "accuracy", "bleu"]
    no_attn_values = [no_attn_metrics[m] for m in metrics]
    attn_values = [attn_metrics[m] for m in metrics]

    # 1. Individual metric comparison (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison: No Attention vs With Attention", fontsize=16, fontweight="bold")

    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        x = ["No Attention", "With Attention"]
        y = [no_attn_values[idx], attn_values[idx]]

        colors = ["#FF6B6B" if metric in ["loss", "perplexity"] else "#4ECDC4"]
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

        ax.set_title(metric.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "comparison_bars.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        no_attn_values,
        width,
        label="No Attention",
        alpha=0.7,
        color="#FF6B6B",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        attn_values,
        width,
        label="With Attention",
        alpha=0.7,
        color="#4ECDC4",
    )

    ax.set_xlabel("Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Values", fontsize=12, fontweight="bold")
    ax.set_title("Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "comparison_grouped.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    improvements = []
    for i, metric in enumerate(metrics):
        if no_attn_values[i] == 0 or abs(no_attn_values[i]) < 1e-10:
            imp = 0  # Avoid division by zero
        elif metric in ["loss", "perplexity"]:
            imp = ((no_attn_values[i] - attn_values[i]) / abs(no_attn_values[i])) * 100
        else:
            imp = ((attn_values[i] - no_attn_values[i]) / abs(no_attn_values[i])) * 100
        improvements.append(imp)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if imp > 0 else "red" for imp in improvements]
    bars = ax.barh([m.capitalize() for m in metrics], improvements, color=colors, alpha=0.7)

    ax.set_xlabel("Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_title("Improvement with Attention Mechanism", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.3)

    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"{imp:+.2f}%",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "improvement_bars.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Validation metrics comparison (bar chart)
    val_metrics = ["val_loss", "val_perplexity"]
    val_no_attn_values = [no_attn_metrics.get(m, 0) for m in val_metrics]
    val_attn_values = [attn_metrics.get(m, 0) for m in val_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(val_metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, val_no_attn_values, width, label="No Attention", alpha=0.8)
    bars2 = ax.bar(x + width / 2, val_attn_values, width, label="With Attention", alpha=0.8)

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax.set_title("Validation Metrics Comparison (Best Checkpoint)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("val_", "").capitalize() for m in val_metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "validation_best_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Created 4 comparison plots in {save_dir}")


def extract_qualitative_examples(report_path, num_examples=20):
    """Extract qualitative examples from report."""
    examples = []

    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("Example"):
                if i + 3 < len(lines):
                    src_line = lines[i + 1].strip()
                    ref_line = lines[i + 2].strip()
                    pred_line = lines[i + 3].strip()

                    if (
                        src_line.startswith("Source:")
                        and ref_line.startswith("Reference:")
                        and pred_line.startswith("Predicted:")
                    ):
                        examples.append(
                            {
                                "source": src_line.replace("Source:", "").strip(),
                                "reference": ref_line.replace("Reference:", "").strip(),
                                "predicted": pred_line.replace("Predicted:", "").strip(),
                            }
                        )
            i += 1

    return examples[:num_examples]


def create_qualitative_comparison(
    no_attn_report, attn_report, save_path="qualitative_comparison.txt"
):
    """Create qualitative comparison."""
    no_attn_examples = extract_qualitative_examples(no_attn_report, num_examples=20)
    attn_examples = extract_qualitative_examples(attn_report, num_examples=20)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("QUALITATIVE COMPARISON: 20 Translation Examples (German → English)\n")
        f.write("=" * 120 + "\n\n")

        for i, (no_attn, attn) in enumerate(zip(no_attn_examples, attn_examples), 1):
            f.write(f"Example {i}:\n")
            f.write("-" * 120 + "\n")
            f.write(f"Source (German):  {no_attn.get('source', 'N/A')}\n")
            f.write(f"Reference:        {no_attn['reference']}\n")
            f.write(f"No Attention:     {no_attn['predicted']}\n")
            f.write(f"With Attention:   {attn['predicted']}\n")
            f.write("\n")

        f.write("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare model results")
    parser.add_argument(
        "--no_attn_report",
        type=str,
        default="comparison_report_no_attention.txt",
        help="Path to no attention model report",
    )
    parser.add_argument(
        "--attn_report",
        type=str,
        default="comparison_report_attention.txt",
        help="Path to attention model report",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--checkpoint_no_attn",
        type=str,
        default="checkpoints_no_attention",
        help="Path to no attention checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_attn",
        type=str,
        default="checkpoints_attention",
        help="Path to attention checkpoint directory",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("ANALYSIS PIPELINE STARTED")
    logger.info("=" * 80)

    # Parse metrics
    logger.info("Parsing evaluation reports...")
    no_attn_metrics = parse_report(args.no_attn_report)
    attn_metrics = parse_report(args.attn_report)

    # Create comparison table
    logger.info("Creating comparison table...")
    create_comparison_table(
        no_attn_metrics, attn_metrics, os.path.join(args.output_dir, "comparison_table.txt")
    )

    # Create comparison plots
    logger.info("Creating comparison plots...")
    create_comparison_plots(no_attn_metrics, attn_metrics, plots_dir)

    # Create validation curves comparison
    logger.info("Creating validation curves comparison...")
    create_validation_comparison_chart(
        args.checkpoint_no_attn,
        args.checkpoint_attn,
        os.path.join(args.output_dir, "validation_comparison.png"),
    )

    # Create qualitative comparison
    logger.info("Creating qualitative comparison (20 examples)...")
    create_qualitative_comparison(
        args.no_attn_report,
        args.attn_report,
        os.path.join(args.output_dir, "qualitative_comparison.txt"),
    )

    # Save metrics as JSON
    logger.info("Saving metrics to JSON...")
    metrics_json = {"no_attention": no_attn_metrics, "with_attention": attn_metrics}

    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"  - Comparison table: comparison_table.txt")
    logger.info(f"  - Qualitative examples: qualitative_comparison.txt (20 examples)")
    logger.info(f"  - Plots: plots/")
    logger.info(f"  - Validation curves: validation_comparison.png")
    logger.info(f"  - Metrics JSON: metrics.json")


if __name__ == "__main__":
    main()
