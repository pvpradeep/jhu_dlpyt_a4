#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 512 \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --dropout 0.5 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode eval \
    --ckpt "$PROJECT_ROOT/outputs/models/best_model_wts.pt"