"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import numpy as np

import logger
from video_datasets import VideoDataset, load_dataset, dataset_split
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN
from train import train
from test import test, get_test_report, get_confusion_matrix
from logger import setup_logger, log_and_print, log_args
from wandb_utils import init_wandb, log_test, finish_wandb

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        -trs/--train_size: Proportion of data to use for training (default 0.7).
        -tss/--test_size: Proportion of data to use for testing (default 0.1).
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
        -ul/--unfreeze_layer: ResNet layer to unfreeze from (e.g., 3 for layer3). -1 freezes the entire backbone.
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('-trs', '--train_size', type=float, default=0.7, help='Train set size')
    parser.add_argument('-tss', '--test_size', type=float, default=0.1, help='Test set size')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', help='3D CNN or LRCN', default='lrcn')
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34', help='2D CNN backbone - options: resnet18, resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('-p', '--pretrained', help='Use pretrained 2D CNN backbone', default=True)
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')

    parser.add_argument('-m', '--mode', type=str, default='train', help="Either 'train' or 'eval'", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-od', '--output_dir', default='./outputs', help='Directory for outputs (models, logs, splits)')
    parser.add_argument('--run_name', type=str, default=None, help='Name for this experiment run (used in logs and W&B)')
    parser.add_argument('-ul', '--unfreeze_layer', type=int, default=-1, help='ResNet layer to unfreeze from (e.g., 3 for layer3). -1 freezes the entire backbone.')

    args = parser.parse_args()
    return args

def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    setup_logger(log_dir=os.path.join(output_dir, "logs"))
    log_args(args)
    init_wandb(args, run_name=args.run_name)
    if args.run_name:
        log_and_print(f"=== Run: {args.run_name} ===")

    frame_dir = args.frame_dir
    tr_size = args.train_size
    ts_size = args.test_size
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available()
                    else 'mps' if torch.backends.mps.is_available()
                    else 'cpu')
    log_and_print(f"Using device: {device}")
    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std)

    # Initialize the model (LRCN)
    model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                 n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone)

    if mode == 'train':
        # Load dataset and split into train/validation/test
        vid_dataset, label_dict = load_dataset(frame_dir)
        tr_split, val_split, ts_split = dataset_split(vid_dataset, tr_size, ts_size)

        # Save the splits for reproducibility and later use in evaluation
        splits = {'train': np.array(tr_split),
                  'val': np.array(val_split),
                  'test': np.array(ts_split)}
        np.save(os.path.join(output_dir, 'splits.npy'), splits)

        # Create PyTorch Datasets and DataLoaders for train and validation
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type)

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.1)
        # Main training procedure
        model.to(device)
        # Resume from checkpoint if provided
        if args.ckpt and os.path.isfile(args.ckpt):
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
            log_and_print(f"run.py Loaded checkpoint from {args.ckpt}")

        # Dynamic freezing/unfreezing based on command line argument
        if args.unfreeze_layer == -1:
            model.freeze_backbone()
            log_and_print("Stage 1: CNN backbone completely frozen.")
        else:
            layer_name = f'layer{args.unfreeze_layer}'
            model.unfreeze_from(layer_name)
            log_and_print(f"Stage 2: CNN backbone unfrozen from {layer_name}.")

        # Differential learning rates for fine-tuning
        if args.unfreeze_layer == -1:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=learning_rate, weight_decay=1e-4)
        else:
            cnn_params = [p for n, p in model.named_parameters() if p.requires_grad and 'base_model' in n]
            head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'base_model' not in n]

            opt = optim.Adam([
                {'params': cnn_params, 'lr': learning_rate * 0.1},  # 10x smaller for CNN
                {'params': head_params, 'lr': learning_rate}        # Base LR for LSTM
            ], weight_decay=1e-4)


        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        optim_model_dir = os.path.join(output_dir, "models")
        os.makedirs(optim_model_dir, exist_ok=True)

        model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device, optim_model_dir, n_epochs)
        best_val = max(acc_hist['val'])
        best_ep = acc_hist['val'].index(best_val) + 1
        log_and_print(f"Training complete. Best val accuracy: {100*best_val:.2f}% at epoch {best_ep}")
        finish_wandb(best_val_accuracy=100*best_val, best_epoch=best_ep)

    elif mode == 'eval':
        # Load saved dataset splits
        splits = np.load(os.path.join(output_dir, 'splits.npy'), allow_pickle=True)
        ts_split = splits.item()['test']
        ts_split = [(sample[0], int(sample[1])) for sample in ts_split]

        # Extract sorted category names and construct the label mapping
        all_cats = sorted([
            d for d in os.listdir(args.frame_dir) 
            if os.path.isdir(os.path.join(args.frame_dir, d)) and not d.startswith('.')
        ])
        labels_dict = {cat: i for i, cat in enumerate(all_cats)}

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type)

        # Load the trained model checkpoint
        if args.ckpt and os.path.isfile(args.ckpt):
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device)
        targets, outputs, accuracy = test(model, dataloaders['test'], device, n_clips=5)

        log_and_print(f"The overall test accuracy is {100 * accuracy:.4f}%.")
        # Optionally, generate a detailed test report or confusion matrix:
        # print(get_test_report(targets, outputs, all_cats))
        # print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))

        # Generate detailed classification report and confusion matrix
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(get_test_report(targets, outputs, all_cats))

        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        print(get_confusion_matrix(targets, outputs, labels_dict, all_cats))
        log_test(100 * accuracy, None)
        finish_wandb()

    else:
        raise ValueError('The mode argument must be either "train" or "eval".')

if __name__ == "__main__":
    args = args_parser()
    main(args)
