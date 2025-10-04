"""
Script to automatically generate README.md for lab assignment.
"""

import os
import yaml


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results(report_path):
    """Load results from comparison report if exists."""
    if not os.path.exists(report_path):
        return None

    results = {}
    with open(report_path, "r") as f:
        content = f.read()

        for line in content.split("\n"):
            if "Test Loss:" in line:
                results["loss"] = line.split(":")[1].strip()
            elif "Test Perplexity:" in line:
                results["perplexity"] = line.split(":")[1].strip()
            elif "Test Accuracy:" in line:
                results["accuracy"] = line.split(":")[1].strip()
            elif "Test BLEU Score:" in line:
                results["bleu"] = line.split(":")[1].strip()
            elif "Test BERTScore (F1):" in line:
                results["bertscore"] = line.split(":")[1].strip()
            elif "Test BLEURT:" in line:
                results["bleurt"] = line.split(":")[1].strip()
            elif "Test COMET:" in line:
                results["comet"] = line.split(":")[1].strip()

    return results if results else None


def get_qualitative_examples(report_path, num_examples=5):
    """Extract qualitative examples from report."""
    if not os.path.exists(report_path):
        return []

    examples = []
    with open(report_path, "r") as f:
        content = f.read()
        lines = content.split("\n")

        i = 0
        while i < len(lines) and len(examples) < num_examples:
            if lines[i].startswith("Example"):
                example = {}
                if i + 1 < len(lines) and "Source:" in lines[i + 1]:
                    example["source"] = lines[i + 1].split("Source:")[1].strip()
                if i + 2 < len(lines) and "Reference:" in lines[i + 2]:
                    example["reference"] = lines[i + 2].split("Reference:")[1].strip()
                if i + 3 < len(lines) and "Predicted:" in lines[i + 3]:
                    example["predicted"] = lines[i + 3].split("Predicted:")[1].strip()
                if example:
                    examples.append(example)
            i += 1

    return examples


def get_model_size_info(checkpoint_path):
    """
    Calculate model size from checkpoint.

    Returns dict with:
    - total_params: total number of parameters
    - size_fp16_mb: size in float16/bfloat16 (MB)
    - size_fp32_mb: size in float32 (MB)
    """
    if not os.path.exists(checkpoint_path):
        return None

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        total_params = 0

        # Count encoder parameters
        if "encoder_state_dict" in checkpoint:
            for param_tensor in checkpoint["encoder_state_dict"].values():
                total_params += param_tensor.numel()

        # Count decoder parameters
        if "decoder_state_dict" in checkpoint:
            for param_tensor in checkpoint["decoder_state_dict"].values():
                total_params += param_tensor.numel()

        # Calculate sizes
        # float16/bfloat16: 2 bytes per parameter
        # float32: 4 bytes per parameter
        size_fp16_mb = (total_params * 2) / (1024 * 1024)
        size_fp32_mb = (total_params * 4) / (1024 * 1024)

        return {
            "total_params": total_params,
            "size_fp16_mb": size_fp16_mb,
            "size_fp32_mb": size_fp32_mb,
        }
    except Exception as e:
        print(f"Warning: Could not load model size from {checkpoint_path}: {e}")
        return None


def get_wandb_info():
    """Extract wandb run info from local wandb runs by reading config and constructing URLs."""
    import glob
    import yaml

    wandb_info = {"has_runs": False, "no_attention": {}, "attention": {}}

    # Get all wandb run directories sorted by modification time (newest first)
    wandb_run_dirs = sorted(
        glob.glob("wandb/run-*"), key=lambda x: os.path.getmtime(x), reverse=True
    )

    for run_dir in wandb_run_dirs:
        try:
            # Extract run_id from directory name (format: run-20251004_214447-yzvv9q4w)
            run_id = run_dir.split("-")[-1]  # Get the last part after last dash

            # Read config to determine model type and project
            config_path = f"{run_dir}/files/config.yaml"
            model_type = None
            project_name = "translation-de-en"  # Default

            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if "model_type" in config:
                        model_type = config["model_type"]["value"]
                    if "wandb_project" in config:
                        project_name = config["wandb_project"]["value"]
            except:
                pass

            # Construct URLs with public access postfix
            # Note: Entity is hardcoded as "Private_1" based on terminal output
            # Update this if your wandb entity is different
            entity = "Private_1"  # CHANGE THIS if your entity is different
            dashboard_url = f"https://wandb.ai/{entity}/{project_name}/runs/{run_id}?nw=nwuser"
            workspace_url = f"https://wandb.ai/{entity}/{project_name}/workspace?nw=nwuser"

            # Store info
            info = {
                "run_id": run_id,
                "dashboard": dashboard_url,
                "workspace": workspace_url,
                "entity": entity,
                "project": project_name,
            }

            if model_type == "no_attention" and not wandb_info["no_attention"]:
                wandb_info["no_attention"] = info
            elif model_type == "attention" and not wandb_info["attention"]:
                wandb_info["attention"] = info
            elif not model_type:
                # If we can't determine model type, assign to the first empty slot
                if not wandb_info["no_attention"]:
                    wandb_info["no_attention"] = info
                elif not wandb_info["attention"]:
                    wandb_info["attention"] = info

            # Stop if we found both
            if wandb_info["no_attention"] and wandb_info["attention"]:
                wandb_info["has_runs"] = True
                break

        except Exception as e:
            continue

    return wandb_info


def generate_readme():
    """Generate comprehensive README with lab assignment results."""

    # Load configurations
    config_no_attn = load_config("config_no_attention.yaml")
    config_attn = load_config("config_attention.yaml")

    # Load results if available
    results_no_attn = load_results("analysis_results/comparison_report_no_attention.txt")
    results_attn = load_results("analysis_results/comparison_report_attention.txt")

    # Get qualitative examples
    examples_no_attn = get_qualitative_examples(
        "analysis_results/comparison_report_no_attention.txt", 12
    )
    examples_attn = get_qualitative_examples("analysis_results/comparison_report_attention.txt", 12)

    # Get wandb info
    wandb_info = get_wandb_info()

    # Get model size info
    model_size_no_attn = get_model_size_info("checkpoints_no_attention/best_model.pth")
    model_size_attn = get_model_size_info("checkpoints_attention/best_model.pth")

    readme_content = f"""# German→English Translation: Attention vs. Non-Attention Models

> **Note:** The dataset is quite large for T4 GPU training. To achieve acceptable translation results efficiently, the original notebooks have been scriptized into production-ready scripts supporting Distributed Data Parallel (DDP) training. 
> 
> **This report is automatically generated** by [`generate_readme.py`](generate_readme.py) after each training run, with results populated from experiments.
> 
> **Translation Direction:** German (input) → English (output) - This allows for easy evaluation since English is more familiar.

---

## Table of Contents

1. [Links to Training Evidence](#1-links-to-training-evidence)
2. [Lab Assignment Results](#2-lab-assignment-results)
   - 2.1 [Model without Attention](#21-model-without-attention)
   - 2.2 [Model with Attention](#22-model-with-attention)
   - 2.3 [Model Size](#23-model-size)
   - 2.4 [Validation Curves](#24-validation-curves)
   - 2.5 [Comparison and Analysis](#25-comparison-and-analysis)
3. [How to Run (For Reviewers)](#3-how-to-run-for-reviewers)
   - 3.1 [Evaluate Pre-trained Models](#31-evaluate-pre-trained-models-from-hugging-face-hub)
   - 3.2 [Full Training Pipeline](#32-full-training-pipeline)
   - 3.3 [Custom Training](#33-custom-training)
4. [Project Structure](#4-project-structure)
5. [Model Architecture](#5-model-architecture)
6. [Configuration](#6-configuration)
7. [Additional Information](#7-additional-information)

---

## 1. Links to Training Evidence

> <sub>**🔴 [AUTO-GENERATED]** Links below are automatically populated from training runs</sub>

This project includes realistic distributed training with experiment tracking:

**Repositories:** <sub>**🔴 [AUTO-GENERATED]**</sub>
- **GitHub:** https://github.com/tuandung222/Translate-EN-DE-Exercise
- **Model Hub:** https://huggingface.co/tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course

**Experiment Tracking (Weights & Biases):** <sub>**🔴 [AUTO-GENERATED]**</sub>
"""

    if wandb_info["has_runs"]:
        if wandb_info["no_attention"]:
            readme_content += f"""
- **Model without Attention:**
  - Dashboard: {wandb_info['no_attention']['dashboard']}
  - Workspace: {wandb_info['no_attention']['workspace']}
"""

        if wandb_info["attention"]:
            readme_content += f"""
- **Model with Attention:**
  - Dashboard: {wandb_info['attention']['dashboard']}
  - Workspace: {wandb_info['attention']['workspace']}
"""

        # Add project overview if we have entity info
        if wandb_info["no_attention"].get("entity") or wandb_info["attention"].get("entity"):
            entity = wandb_info["no_attention"].get("entity") or wandb_info["attention"].get(
                "entity"
            )
            project = wandb_info["no_attention"].get("project") or wandb_info["attention"].get(
                "project"
            )
            readme_content += f"""
- **Project Overview:** https://wandb.ai/{entity}/{project}?nw=nwuser

> All wandb links include `?nw=nwuser` for public access without login.
"""
    else:
        readme_content += """
*Wandb experiment links will appear here after training completes with `--use_wandb` flag.*
"""

    readme_content += """
**What's Tracked:**
- Training/validation loss curves
- Perplexity over time
- Learning rate schedule
- Test set evaluation metrics (accuracy, BLEU)
- Model architecture and hyperparameters
- Hardware utilization (GPU, memory)

---

## Lab Assignment Results
"""

    readme_content += f"""
### 2.1 Model without Attention

**Architecture:** {config_no_attn["num_layers"]}-layer GRU encoder-decoder, no attention mechanism

**Training:** {config_no_attn["n_epochs"]} epochs, batch {config_no_attn["batch_size"]}, hidden {config_no_attn["hidden_size"]}, bfloat16 precision

**Results:** <sub>**🔴 [AUTO-GENERATED]**</sub>

"""

    if results_no_attn:
        readme_content += f"""| Metric | Score |
|--------|-------|
| Accuracy | {results_no_attn['accuracy']} |
| BLEU | {results_no_attn['bleu']} |
| BERTScore (F1) | {results_no_attn.get('bertscore', 'N/A')} |
| BLEURT | {results_no_attn.get('bleurt', 'N/A')} |
| COMET | {results_no_attn.get('comet', 'N/A')} |
| Test Loss | {results_no_attn['loss']} |
"""
    else:
        readme_content += "\n*Training in progress...*\n"

    readme_content += f"""
---

### 2.2 Model with Attention

**Architecture:** {config_attn["num_layers"]}-layer GRU encoder-decoder with Bahdanau attention

**Training:** {config_attn["n_epochs"]} epochs, batch {config_attn["batch_size"]}, hidden {config_attn["hidden_size"]}, bfloat16 precision

**Results:** <sub>**🔴 [AUTO-GENERATED]**</sub>

"""

    if results_attn:
        readme_content += f"""| Metric | Score |
|--------|-------|
| Accuracy | {results_attn['accuracy']} |
| BLEU | {results_attn['bleu']} |
| BERTScore (F1) | {results_attn.get('bertscore', 'N/A')} |
| BLEURT | {results_attn.get('bleurt', 'N/A')} |
| COMET | {results_attn.get('comet', 'N/A')} |
| Test Loss | {results_attn['loss']} |
"""
    else:
        readme_content += "\n*Training in progress...*\n"

    readme_content += """
---

### 2.3 Model Size

<sub>**🔴 [AUTO-GENERATED]** Model sizes calculated from checkpoints:</sub>

"""

    # Add model size comparison table
    if model_size_no_attn and model_size_attn:
        readme_content += f"""| Model | Parameters | Size (bfloat16) | Size (float32) |
|-------|------------|-----------------|----------------|
| No Attention | {model_size_no_attn['total_params']:,} | {model_size_no_attn['size_fp16_mb']:.2f} MB | {model_size_no_attn['size_fp32_mb']:.2f} MB |
| With Attention | {model_size_attn['total_params']:,} | {model_size_attn['size_fp16_mb']:.2f} MB | {model_size_attn['size_fp32_mb']:.2f} MB |

**Notes:**
- Training uses **bfloat16** mixed precision for efficiency
- Parameter count includes encoder + decoder weights
- Attention model has additional parameters for attention mechanism (~{((model_size_attn['total_params'] - model_size_no_attn['total_params']) / model_size_no_attn['total_params'] * 100):.1f}% more)
"""
    else:
        readme_content += "\n*Model size information will be available after training.*\n"

    readme_content += """
---

### 2.4 Validation Curves

<sub>**🔴 [AUTO-GENERATED]** Training progress comparison showing both models:</sub>

![Validation Comparison](analysis_results/validation_comparison.png)

*This chart compares validation loss and perplexity over training steps for both models, demonstrating convergence speed and final performance.*

---

### 2.5 Comparison and Analysis

**Quantitative Comparison:** <sub>**🔴 [AUTO-GENERATED]**</sub>
"""

    if results_no_attn and results_attn:
        # Extract all metrics
        no_attn_acc = float(results_no_attn["accuracy"])
        attn_acc = float(results_attn["accuracy"])
        no_attn_bleu = float(results_no_attn["bleu"])
        attn_bleu = float(results_attn["bleu"])

        # Calculate improvements with safe division
        def safe_improvement(baseline, new_value):
            if abs(baseline) < 1e-10:  # baseline is ~0
                return "N/A (baseline≈0)"
            improvement = (new_value - baseline) / abs(baseline) * 100
            return f"+{improvement:.1f}%"

        acc_improvement = safe_improvement(no_attn_acc, attn_acc)
        bleu_improvement = safe_improvement(no_attn_bleu, attn_bleu)

        readme_content += f"""
| Metric | No Attention | With Attention | Improvement |
|--------|--------------|----------------|-------------|
| Accuracy | {results_no_attn['accuracy']} | {results_attn['accuracy']} | {acc_improvement} |
| BLEU | {results_no_attn['bleu']} | {results_attn['bleu']} | {bleu_improvement} |
"""

        # Add advanced metrics if available
        if "bertscore" in results_no_attn and "bertscore" in results_attn:
            no_attn_bert = float(results_no_attn.get("bertscore", "0"))
            attn_bert = float(results_attn.get("bertscore", "0"))
            bert_improvement = safe_improvement(no_attn_bert, attn_bert)
            readme_content += f"| BERTScore (F1) | {results_no_attn.get('bertscore', 'N/A')} | {results_attn.get('bertscore', 'N/A')} | {bert_improvement} |\n"

        if "bleurt" in results_no_attn and "bleurt" in results_attn:
            no_attn_bleurt = float(results_no_attn.get("bleurt", "0"))
            attn_bleurt = float(results_attn.get("bleurt", "0"))
            # BLEURT is often 0 (not installed), skip if both are 0
            if not (abs(no_attn_bleurt) < 1e-10 and abs(attn_bleurt) < 1e-10):
                bleurt_improvement = safe_improvement(no_attn_bleurt, attn_bleurt)
                readme_content += f"| BLEURT | {results_no_attn.get('bleurt', 'N/A')} | {results_attn.get('bleurt', 'N/A')} | {bleurt_improvement} |\n"

        if "comet" in results_no_attn and "comet" in results_attn:
            no_attn_comet = float(results_no_attn.get("comet", "0"))
            attn_comet = float(results_attn.get("comet", "0"))
            comet_improvement = safe_improvement(no_attn_comet, attn_comet)
            readme_content += f"| COMET | {results_no_attn.get('comet', 'N/A')} | {results_attn.get('comet', 'N/A')} | {comet_improvement} |\n"

        # Add test loss comparison
        if "loss" in results_no_attn and "loss" in results_attn:
            no_attn_loss = float(results_no_attn["loss"])
            attn_loss = float(results_attn["loss"])
            # For loss, lower is better, so we calculate reduction instead
            loss_reduction = (
                ((no_attn_loss - attn_loss) / no_attn_loss * 100) if no_attn_loss > 0 else 0
            )
            readme_content += f"| Test Loss | {results_no_attn['loss']} | {results_attn['loss']} | {loss_reduction:.1f}% reduction |\n"
    else:
        readme_content += "\n*Awaiting training results...*\n"

    readme_content += """
<sub>**🔴 [AUTO-GENERATED]** **Analysis:**</sub>
"""

    if results_no_attn and results_attn:
        # Determine which is better
        attn_accuracy = float(results_attn["accuracy"])
        no_attn_accuracy = float(results_no_attn["accuracy"])
        attn_bleu = float(results_attn["bleu"])
        no_attn_bleu = float(results_no_attn["bleu"])

        if attn_accuracy > no_attn_accuracy and attn_bleu > no_attn_bleu:
            readme_content += (
                "\nAttention mechanism performs better on both accuracy and BLEU score.\n"
            )
        elif attn_accuracy > no_attn_accuracy or attn_bleu > no_attn_bleu:
            readme_content += "\nAttention mechanism shows improvement in translation quality.\n"
        else:
            readme_content += "\nBoth models show similar performance.\n"
    else:
        readme_content += "\n*Analysis will be generated after training completes.*\n"

    readme_content += """
<sub>**🔴 [AUTO-GENERATED]** **Qualitative Examples:**</sub>
"""

    if examples_no_attn and examples_attn:
        readme_content += """
| Source (German) | Reference (English) | No Attention | With Attention |
|----------------|--------------------|--------------|--------------------|
"""
        for i in range(min(12, len(examples_no_attn), len(examples_attn))):
            src = examples_no_attn[i].get("source", "N/A")
            ref = examples_no_attn[i]["reference"]
            pred_no_attn = examples_no_attn[i]["predicted"]
            pred_attn = examples_attn[i]["predicted"]
            readme_content += f"| {src} | {ref} | {pred_no_attn} | {pred_attn} |\n"
    else:
        readme_content += "\n*Examples will be generated after training.*\n"

    readme_content += """
---

## 3. How to Run (For Reviewers)

### 3.1 Evaluate Pre-trained Models from Hugging Face Hub

Download and evaluate our trained models without training from scratch:

```bash
# Install dependencies
pip install -r requirements.txt

# Evaluate model with attention
python predict.py --from_hub --model_type attention --sentence "hello world"

# Evaluate model without attention
python predict.py --from_hub --model_type no_attention --sentence "ich bin gut"

# Test on random examples from test set
python predict.py --from_hub --model_type attention --num_examples 10
```

**Model Hub Structure:**
- `tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course/no_attention/best_model.pth`
- `tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course/attention/best_model.pth`

### 3.2 Full Training Pipeline

Train both models from scratch and reproduce all results:

```bash
# Install dependencies
pip install -r requirements.txt

# Run full automated pipeline (train, evaluate, upload, commit)
# Uses 4 GPUs by default (configurable via CUDA_VISIBLE_DEVICES)
./distributed_run_full_pipeline.sh 4 true
```

This will:
1. Train both models (no_attention and attention)
2. Evaluate on test set
3. Generate comparison reports and visualizations
4. Update this README with results
5. Push checkpoints to Hugging Face Hub
6. Commit and push to GitHub

### 3.3 Custom Training

```bash
# Single GPU training
python train.py --config config_attention.yaml --use_wandb

# Multi-GPU distributed training (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py \\
    --config config_attention.yaml --use_wandb
```

---

## 4. Project Structure

```
translate-de-en/
├── Core Model & Data
│   ├── model.py                      # Model architectures (Encoder, Decoder, Attention)
│   ├── data_loader.py                # Data loading with HuggingFace datasets
│   └── utils.py                      # Unified utilities (metrics, sampling, distributed, etc.)
├── Training & Evaluation
│   ├── train.py                      # Main training loop with DDP and mixed precision
│   └── evaluation.py                 # Model evaluation functions
├── Inference & Analysis
│   ├── predict.py                    # Inference with nucleus sampling
│   ├── analysis.py                   # Compare models and generate plots
│   └── generate_readme.py            # Auto-generate this README
├── Configuration & Scripts
│   ├── config_no_attention.yaml      # Hyperparameters for model without attention
│   ├── config_attention.yaml         # Hyperparameters for model with attention
│   ├── distributed_run_full_pipeline.sh  # Main automated training pipeline
│   ├── push_to_hub.py                # Upload models to HuggingFace Hub
│   └── test_predict.sh               # Quick prediction test script
├── Dependencies & Configuration
│   ├── requirements.txt              # Python dependencies
│   └── pyproject.toml                # Black formatter configuration
├── Data & Outputs
│   ├── data/                         # Dataset directory (auto-downloaded)
│   ├── checkpoints_attention/        # Attention model checkpoints + vocab
│   ├── checkpoints_no_attention/     # No-attention model checkpoints + vocab
│   └── analysis_results/             # All analysis outputs and reports
│       ├── comparison_report_*.txt   # Detailed evaluation reports
│       ├── comparison_table.txt      # Side-by-side metrics comparison
│       ├── qualitative_comparison.txt # Translation examples comparison
│       ├── metrics.json              # JSON metrics for programmatic access
│       ├── validation_comparison.png # Validation curves plot
│       └── plots/                    # Additional visualization plots
├── Notebooks & Archive
│   ├── notebooks/                    # Original Jupyter notebooks
│   └── archive_docs/                 # Archived documentation files
└── Cache & Temp
    ├── dataset_cache/                # HuggingFace datasets cache
    ├── wandb/                        # Weights & Biases logs
    └── __pycache__/                  # Python bytecode cache
```

**Key Features:**
- **Clean Organization:** All reports in `analysis_results/`, no clutter in root
- **Unified Utilities:** All helper functions consolidated in `utils.py`
- **Separate Checkpoints:** Each model type has its own checkpoint directory
- **Automated Pipeline:** Single script runs training, evaluation, and documentation

---

## 5. Model Architecture

<sub>**🔴 [AUTO-GENERATED]** Architecture specs populated from config files.</sub>
"""

    readme_content += f"""
### Encoder-Decoder Architecture

This project uses **Residual Stacked GRU** with **Layer Normalization** to train deep networks effectively.

**Implementation Classes** (see `model.py`):
- `ResidualStackedGRUEncoder` - Encoder with residual connections and layer normalization
- `ResidualStackedGRUDecoder` - Decoder without attention
- `ResidualStackedGRUAttnDecoder` - Decoder with Bahdanau attention mechanism

**Key Features:**
- **Residual Connections:** Enable gradient flow through deep networks (4-8 layers)
- **Layer Normalization:** Stabilizes training and improves convergence
- **Xavier Initialization:** Ensures proper gradient flow from the start

### Model Configuration

**Encoder (Shared):**
- Type: Residual Stacked GRU
- Layers: {config_no_attn["num_layers"]}
- Hidden Size: {config_no_attn["hidden_size"]}
- Dropout: {config_no_attn["dropout"]}

**Decoder (No Attention):**
- Type: Residual Stacked GRU
- Layers: {config_no_attn["num_layers"]}
- Hidden Size: {config_no_attn["hidden_size"]}
- Dropout: {config_no_attn["dropout"]}

**Decoder (With Attention):**
- Type: Residual Stacked GRU + Bahdanau Attention
- Layers: {config_attn["num_layers"]}
- Hidden Size: {config_attn["hidden_size"]}
- Attention Dim: {config_attn["hidden_size"] // 2}
- Dropout: {config_attn["dropout"]}

### Training Configuration
- **Precision:** bfloat16
- **Optimizer:** AdamW8bit (bitsandbytes)
- **Scheduler:** CosineAnnealingLR
- **Gradient Clipping:** Max norm = {config_no_attn["max_grad_norm"]}
- **Distributed:** PyTorch DDP
- **Teacher Forcing:** 100% during training

### Evaluation Strategy

**Validation (during training):**
- Uses validation loss and perplexity
- Teacher forcing for fast computation
- Evaluated every 1/2 epoch for monitoring convergence

**Test Set (final evaluation):**
- Auto-regressive generation with nucleus sampling for qualitative results
- Nucleus sampling parameters (optimized for translation quality):
  - `top_k`: 20 (consider top-20 tokens)
  - `top_p`: 0.6 (nucleus threshold)
  - `temperature`: 0.3 (low temperature for more deterministic, accurate translations)
  - `repetition_penalty`: 1.05 (discourage repetition)
  - `max_length`: 64 (maximum generation length)
- Limited to 512 instances for reasonable computation time
- Evaluation metrics:
  - **Accuracy**: Token-level matching with reference
  - **BLEU**: Standard n-gram overlap metric
  - **BERTScore**: Semantic similarity using DistilBERT embeddings
  - **BLEURT**: Learned evaluation metric (BLEURT-20-D3)
  - **COMET**: Translation quality estimation (wmt22-cometkiwi-da)
- Qualitative translation examples for human assessment
"""

    readme_content += """

---

## 6. Configuration

Models are configured via YAML files. Edit these files to change hyperparameters:

### `config_no_attention.yaml` / `config_attention.yaml`

<sub>**🔴 [AUTO-GENERATED]** Configuration values below are automatically loaded from YAML files.</sub>
"""

    readme_content += f"""
```yaml
model_type: attention              # 'attention' or 'no_attention'
hidden_size: {config_attn["hidden_size"]}                   # Hidden dimension
num_layers: {config_attn["num_layers"]}                     # Number of GRU layers
dropout: {config_attn["dropout"]}                       # Dropout probability
max_length: {config_attn["max_length"]}                    # Max sequence length

n_epochs: {config_attn["n_epochs"]}                        # Training epochs
batch_size: {config_attn["batch_size"]}                    # Batch size (per GPU)
learning_rate: {config_attn["learning_rate"]}             # Initial learning rate
max_grad_norm: {config_attn["max_grad_norm"]}                  # Gradient clipping

data_source: local                 # 'local' or 'tatoeba'
num_workers: {config_attn["num_workers"]}                   # DataLoader workers

save_dir: checkpoints_attention    # Checkpoint directory
wandb_project: translation-de-en   # Wandb project name
```
"""

    readme_content += """

**Key Parameters:**
- `num_layers`: Increase for more capacity (8 → 16 layers)
- `batch_size`: Larger batches for faster training (GPU memory permitting)
- `n_epochs`: More epochs for lower loss (1 → 50 for production)
- `hidden_size`: Model capacity (256 → 768 → 1024)

---

## 7. Additional Information

### Automatic Features

This project includes full automation:

**Auto-Training Pipeline:**
- Multi-GPU distributed training (DDP)
- Automatic checkpoint saving (best validation loss)
- Vocabulary persistence (pkl files)
- Validation translation sampling (track improvement)

**Auto-Evaluation:**
- Test set evaluation after training
- BLEU score, accuracy, perplexity calculation
- Qualitative translation examples
- Comparison table generation with plots

**Auto-Documentation:**
- This README regenerates after each training run
- Results automatically populated from experiment reports
- Qualitative examples dynamically inserted
- Wandb experiment links auto-detected

**Auto-Upload:**
- Models pushed to Hugging Face Hub: `tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course`
- Model cards generated with training details
- Vocabulary files uploaded for inference

**Auto-Git:**
- Results committed to repository
- README and reports pushed to GitHub
- Training metrics in commit message

### Links

- **GitHub:** https://github.com/tuandung222/Translate-EN-DE-Exercise
- **Model Hub:** https://huggingface.co/tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course
- **Dataset:** German-English translation pairs (~150K pairs)
- **Wandb Project:** https://wandb.ai/TuanDung111/translation-de-en

### Code Quality

- **Formatting:** Code is formatted with `black` (line length = 100)
- **Logging:** Minimal, meaningful logs only (not verbose)
- **Documentation:** This README is auto-generated after each training run
- **Results:** Dynamic metrics populated from actual training experiments
- **Testing:** All division operations protected against zero values
"""

    # Write both README.md and REPORT_EXERCISE.md with same content
    with open("README.md", "w") as f:
        f.write(readme_content)

    with open("REPORT_EXERCISE.md", "w") as f:
        f.write(readme_content)

    print("README.md and REPORT_EXERCISE.md generated successfully!")
    print()
    print("Contents:")
    print("  1. Model without Attention (implementation + results)")
    print("  2. Model with Attention (implementation + results)")
    print("  3. Comparison and Analysis")
    print("  4. Project Structure")
    print("  5. How to Run")
    print()
    if results_no_attn and results_attn:
        print("✓ Results populated from training reports")
        print("✓ Qualitative examples included")
    else:
        print("→ Train models to populate results")
        print("→ Run: ./distributed_run_full_pipeline.sh 4 true")


if __name__ == "__main__":
    generate_readme()
