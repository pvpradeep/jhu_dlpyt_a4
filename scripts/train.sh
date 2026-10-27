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
    --learning_rate 1e-4 \
    --dropout 0.5 \
    --fr_per_vid 32 \
    --batch_size 8 \
    --mode train \
    --ckpt "$PROJECT_ROOT/outputs/models/v2_frfixed_frozen_rs34_best_model_wts.pt" \
    --run_name "v2_-unfrozen-l3-r34-lstm256" \
    --n_epochs 30
