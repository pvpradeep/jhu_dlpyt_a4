#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames_contig_5clips" \
    --flow_dir "$PROJECT_ROOT/data/UCF50_flow_contig_5clips" \
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
    --batch_size 32 \
    --mode eval_dual \
    --ckpt_rgb "$PROJECT_ROOT/outputs/models/v3_stg2-frozen-contig5-r50_best_model_wts.pt" \
    --ckpt_flow "$PROJECT_ROOT/outputs/models/v3_flow_stg2-l3unfrozen-contig5-r50-lstm256_best_model_wts.pt" \
    --alpha 0.5 \
    --run_name "v3_dual_stream_eval_ep4"