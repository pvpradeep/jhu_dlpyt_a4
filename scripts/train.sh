#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$PROJECT_ROOT/src/run.py" \
    --cnn_backbone resnet50 \
    --rnn_hidden_size 512 \
    --frame_dir $PROJECT_ROOT/data/UCF50_frames \
    --output_dir "$PROJECT_ROOT/outputs" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --learning_rate 1e-4 \
    --dropout 0.5 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode 'train' \
    --ckpt "$PROJECT_ROOT/outputs/models/stg1_best_model_wts.pt" \
    --finetune \
    --n_epochs 20

# STage 2
#-     --learning_rate 1e-3 \
#+     --learning_rate 1e-4 \
#      --dropout 0.5 \
#      --fr_per_vid 16 \
#      --batch_size 16 \
#      --mode 'train' \
#-     --n_epochs 50
#+     --n_epochs 20 \
#+     --ckpt "$PROJECT_ROOT/outputs/models/best_model_wts.pt" \
#+     --finetune
#