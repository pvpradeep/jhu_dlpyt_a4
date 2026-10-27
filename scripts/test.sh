#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames_contig_5clips" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 256 \
    --rnn_n_layers 2 \
    --n_classes 50 \
    --dropout 0.5 \
    --fr_per_vid 16 \
    --batch_size 32  \
    --mode eval \
    --ckpt "$PROJECT_ROOT/outputs/models/v3_stg2-frozen-contig5-r50_best_model_wts.pt" \
    --run_name "v3Test_stg2-l3unfrozen-contig5-r50-lstm256"