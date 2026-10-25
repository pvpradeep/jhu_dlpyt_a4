#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --frame_dir "$PROJECT_ROOT/data/UCF50_frames" \
    --output_dir "$PROJECT_ROOT/outputs" \
    --ckpt "$PROJECT_ROOT/outputs/models/best_model_wts.pt" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval
