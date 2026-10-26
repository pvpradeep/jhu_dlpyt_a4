#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames_32" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet34 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --dropout 0.5 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --mode eval \
    --ckpt "$PROJECT_ROOT/outputs/models/best_model_wts.pt"