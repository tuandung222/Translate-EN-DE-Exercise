#!/bin/bash

# Distributed training pipeline for training both models and comparing results
# This script uses GPUs 2,3,4,5 by default (4 GPUs total)
# 
# Usage: ./distributed_run_full_pipeline.sh <num_gpus> <use_wandb>
# Example: ./distributed_run_full_pipeline.sh 4 true

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_GPUS=${1:-8}
USE_WANDB=${2:-true}



# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
# NUM_GPUS=${1:-4}
# USE_WANDB=${2:-true}


# Config files
CONFIG_NO_ATTENTION="config_no_attention.yaml"
CONFIG_ATTENTION="config_attention.yaml"

echo "====================================="
echo "Distributed Training Pipeline"
echo "====================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES (using $NUM_GPUS GPUs)"
echo "Config (no attention): $CONFIG_NO_ATTENTION"
echo "Config (attention): $CONFIG_ATTENTION"
echo "W&B logging: $USE_WANDB"
echo "====================================="
echo ""

if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
else
    WANDB_FLAG=""
fi

echo "Training model without attention"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --config $CONFIG_NO_ATTENTION \
        $WANDB_FLAG
else
    python train.py \
        --config $CONFIG_NO_ATTENTION \
        $WANDB_FLAG
fi

echo ""
echo "Training model with attention"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --config $CONFIG_ATTENTION \
        $WANDB_FLAG
else
    python train.py \
        --config $CONFIG_ATTENTION \
        $WANDB_FLAG
fi

echo ""
echo "Running analysis"
python analysis.py \
    --no_attn_report analysis_results/comparison_report_no_attention.txt \
    --attn_report analysis_results/comparison_report_attention.txt \
    --output_dir analysis_results

echo ""
echo "Updating README.md and REPORT_EXERCISE.md with results"
python generate_readme.py

echo ""
echo "Pushing models to Hugging Face Hub"
python push_to_hub.py --checkpoint_dir checkpoints_no_attention --model_type no_attention
python push_to_hub.py --checkpoint_dir checkpoints_attention --model_type attention

echo ""
echo "Formatting code with black before commit"
black *.py || echo "Black formatting completed (some files may already be formatted)"

echo ""
echo "Pushing to GitHub"
git add .
git add README.md REPORT_EXERCISE.md analysis_results/
git commit -m "Training complete: Update results

- Model without attention: $(grep 'Test BLEU' analysis_results/comparison_report_no_attention.txt | awk '{print $4}')
- Model with attention: $(grep 'Test BLEU' analysis_results/comparison_report_attention.txt | awk '{print $4}')
- Auto-generated report and README
- Code formatted with black" || echo "No changes to commit"

git push origin main || echo "Push failed - authenticate manually: git push origin main"

echo ""
echo "==========================================
echo "Pipeline completed!"
echo "=========================================="
echo ""
echo "Results: analysis_results/"
echo "Reports: README.md and REPORT_EXERCISE.md updated with training results"
echo "GitHub: https://github.com/tuandung222/Translate-EN-DE-Exercise"
echo ""
