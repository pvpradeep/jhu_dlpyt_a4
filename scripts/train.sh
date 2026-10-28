#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames_contig_5clips" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 512 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --learning_rate 4e-4 \
    --dropout 0.5 \
    --fr_per_vid 16 \
    --batch_size 32 \
    --mode train \
    --ckpt "$PROJECT_ROOT/outputs/models/v4_stg1-l3frozen-contig5-r50-lstm512_best_model_wts.pt" \
    --run_name "v4_stg2-l3unfrozen-contig5-r50-lstm512" \
    --unfreeze_layer 3 \
    --n_epochs 30
