#!/bin/bash
# Test script to verify predict.py works with downloaded models

echo "=========================================="
echo "Testing predict.py with Hugging Face Hub"
echo "=========================================="
echo ""

echo "Test 1: Single sentence translation (Attention model)"
echo "------------------------------------------------------"
python predict.py \
    --from_hub \
    --model_type attention \
    --hidden_size 512 \
    --num_layers 5 \
    --use_residual \
    --sentence "ich bin ein student"

echo ""
echo ""
echo "Test 2: Single sentence translation (No Attention model)"
echo "---------------------------------------------------------"
python predict.py \
    --from_hub \
    --model_type no_attention \
    --hidden_size 512 \
    --num_layers 5 \
    --use_residual \
    --sentence "ich bin ein student"

echo ""
echo ""
echo "Test 3: Multiple examples (Attention model)"
echo "--------------------------------------------"
python predict.py \
    --from_hub \
    --model_type attention \
    --hidden_size 512 \
    --num_layers 5 \
    --use_residual \
    --num_examples 5

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
