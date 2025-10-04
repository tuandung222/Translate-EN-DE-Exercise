"""
Push trained models to Hugging Face Hub.
"""

import os
import torch
import pickle
import argparse
from huggingface_hub import HfApi, create_repo


def push_model_to_hub(
    checkpoint_dir,
    model_type,
    repo_id="tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course",
):
    """
    Push model checkpoint and vocab to Hugging Face Hub.

    Args:
        checkpoint_dir: Directory containing best_model.pth and vocab files
        model_type: 'no_attention' or 'attention'
        repo_id: Hugging Face Hub repository ID
    """
    print(f"\nPushing {model_type} model to Hugging Face Hub...")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"✓ Repository {repo_id} ready")
    except Exception as e:
        print(f"Note: {e}")

    # Files to upload
    files_to_upload = [
        (os.path.join(checkpoint_dir, "best_model.pth"), f"{model_type}/best_model.pth"),
        (os.path.join(checkpoint_dir, "input_lang.pkl"), f"{model_type}/input_lang.pkl"),
        (os.path.join(checkpoint_dir, "output_lang.pkl"), f"{model_type}/output_lang.pkl"),
    ]

    # Upload each file
    for local_path, hub_path in files_to_upload:
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=hub_path,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"✓ Uploaded {hub_path}")
            except Exception as e:
                print(f"✗ Failed to upload {hub_path}: {e}")
        else:
            print(f"✗ File not found: {local_path}")

    # Create model card if it doesn't exist
    model_card_path = os.path.join(checkpoint_dir, "README.md")
    if not os.path.exists(model_card_path):
        # Read results from comparison report
        report_path = f"analysis_results/comparison_report_{model_type}.txt"
        results = {}
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                content = f.read()
                for line in content.split("\n"):
                    if "Test Loss:" in line:
                        results["loss"] = line.split(":")[1].strip()
                    elif "Test Accuracy:" in line:
                        results["accuracy"] = line.split(":")[1].strip()
                    elif "Test BLEU Score:" in line:
                        results["bleu"] = line.split(":")[1].strip()

        model_card = f"""---
language:
- de
- en
license: mit
tags:
- translation
- seq2seq
- gru
- {'attention' if model_type == 'attention' else 'no-attention'}
datasets:
- custom
metrics:
- bleu
- accuracy
---

# German-English Translation Model ({'With Attention' if model_type == 'attention' else 'Without Attention'})

Seq2seq model for German to English translation.

## Model Details

- **Architecture:** 16-layer Stacked GRU Encoder-Decoder {'with Bahdanau Attention' if model_type == 'attention' else 'without Attention'}
- **Hidden Size:** 768
- **Training Precision:** bfloat16
- **Optimizer:** AdamW8bit
- **Dataset:** German-English translation pairs

## Performance

{f'''
- **Accuracy:** {results.get('accuracy', 'N/A')}
- **BLEU Score:** {results.get('bleu', 'N/A')}
- **Loss:** {results.get('loss', 'N/A')}
''' if results else '(Training in progress)'}

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
model_path = hf_hub_download(
    repo_id="tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course",
    filename="{model_type}/best_model.pth"
)

# Load model
checkpoint = torch.load(model_path)
# Use with predict.py for inference
```

## Training

Model trained on custom German-English dataset with distributed training (DDP) on 4 GPUs.

**Repository:** https://github.com/tuandung222/Translate-EN-DE-Exercise

## Citation

```bibtex
@misc{{de-en-translation-{model_type},
  title={{German-English Translation {'with' if model_type == 'attention' else 'without'} Attention}},
  author={{Lab Assignment}},
  year={{2025}},
  url={{https://huggingface.co/tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course}}
}}
```
"""

        with open(model_card_path, "w") as f:
            f.write(model_card)

        # Upload model card
        try:
            api.upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo=f"{model_type}/README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"✓ Uploaded model card")
        except Exception as e:
            print(f"✗ Failed to upload model card: {e}")

    print(f"✓ Model {model_type} pushed to Hub: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["no_attention", "attention"]
    )
    parser.add_argument(
        "--repo_id", type=str, default="tuandunghcmut/Exercise-Translate-EN-DE-LLM-Course"
    )

    args = parser.parse_args()

    push_model_to_hub(
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
