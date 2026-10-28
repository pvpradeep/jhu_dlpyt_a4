#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_flow_contig_5clips" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --learning_rate 4e-4 \
    --dropout 0.5 \
    --fr_per_vid 16 \
    --batch_size 32 \
    --mode train \
    --run_name "v3_flow_stg2-l3unfrozen-contig5-r50-lstm256" \
    --ckpt "$PROJECT_ROOT/outputs/models/v3_flow_stg1-frozen-contig5-r50-lstm256_best_model_wts.pt" \
    --unfreeze_layer 3 \
    --n_epochs 30

