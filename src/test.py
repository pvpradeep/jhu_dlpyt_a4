"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def test(model, dataloader, device, n_clips=5):
    """
    Evaluate the model on the test dataset using temporal averaging across multiple clips.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    with torch.no_grad():
        all_logits, all_targets = [], []
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            all_logits.append(output.cpu())
            all_targets.append(y_batch.cpu())

        # Combine all batches
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
       
        # Group sequentially by n_clips and average the logits
        num_videos = len(all_logits) // n_clips
        avg_logits = all_logits[:num_videos * n_clips].view(num_videos, n_clips, -1).mean(dim=1)
        video_targets = all_targets[:num_videos * n_clips].view(num_videos, n_clips)[:, 0]
       
        # Compute final predictions from the averaged video logits
        preds = avg_logits.argmax(dim=1)
        accuracy = preds.eq(video_targets).sum().item() / float(num_videos)
    
    return video_targets.tolist(), preds.tolist(), accuracy
    

import torch
import numpy as np

def test_dual_stream(model_rgb, model_flow, dataloader_rgb, dataloader_flow, device, n_clips=5, alpha_step=0.05):
    """
    Evaluates a Decoupled Two-Stream LRCN pipeline using soft-voting logit ensembling.
    Collects raw stream logits once, averages across video clips, and performs an instant alpha sweep.
    
    Args:
        model_rgb (nn.Module): Trained RGB stream LRCN model.
        model_flow (nn.Module): Trained Optical Flow stream LRCN model.
        dataloader_rgb (DataLoader): Test dataloader for RGB frames.
        dataloader_flow (DataLoader): Test dataloader for Optical Flow maps.
        device (torch.device): 'mps', 'cuda', or 'cpu'.
        n_clips (int): Number of contiguous clips per video for temporal averaging.
        alpha_step (float): Step size for grid searching alpha from 0.0 to 1.0.
        
    Returns:
        tuple: (video_targets_np, best_fused_outputs_np, best_accuracy, best_alpha)
    """
    model_rgb.eval()
    model_flow.eval()

    clip_logits_rgb = []
    clip_logits_flow = []
    clip_targets = []

    # 1. Single Forward Pass: Collect raw un-fused logits
    with torch.no_grad():
        for (batch_rgb, batch_flow) in zip(dataloader_rgb, dataloader_flow):
            inputs_rgb, targets_rgb = batch_rgb[0].to(device), batch_rgb[1].to(device)
            inputs_flow, targets_flow = batch_flow[0].to(device), batch_flow[1].to(device)

            # Sanity check temporal alignment
            assert torch.equal(targets_rgb, targets_flow), "RGB and Flow dataloaders are out of sync!"

            logits_rgb = model_rgb(inputs_rgb)
            logits_flow = model_flow(inputs_flow)

            clip_logits_rgb.append(logits_rgb.cpu())
            clip_logits_flow.append(logits_flow.cpu())
            clip_targets.append(targets_rgb.cpu())

    # Concatenate clip-level tensors
    clip_logits_rgb = torch.cat(clip_logits_rgb, dim=0) # shape: (num_clips, num_classes), RGB logits
    clip_logits_flow = torch.cat(clip_logits_flow, dim=0) # shape: (num_clips, num_classes), Flow logits
    clip_targets = torch.cat(clip_targets, dim=0) # shape: (num_clips,), Ground truth labels for each clip

    # 2. Video-Level Temporal Averaging per Stream
    n_samples = clip_targets.size(0)
    video_logits_rgb = []
    video_logits_flow = []
    video_targets = []

    for i in range(0, n_samples, n_clips): # Iterate over clips in steps of n_clips=5
        vid_rgb = clip_logits_rgb[i : i + n_clips].mean(dim=0)  # Average logits across clips for RGB stream
        vid_flow = clip_logits_flow[i : i + n_clips].mean(dim=0)# Average logits across clips for Flow stream
        
        video_logits_rgb.append(vid_rgb)
        video_logits_flow.append(vid_flow)
        video_targets.append(clip_targets[i])

    video_logits_rgb = torch.stack(video_logits_rgb)
    video_logits_flow = torch.stack(video_logits_flow)
    video_targets = torch.stack(video_targets)

    # Standalone stream accuracies
    acc_rgb = (torch.argmax(video_logits_rgb, dim=1) == video_targets).float().mean().item()
    acc_flow = (torch.argmax(video_logits_flow, dim=1) == video_targets).float().mean().item()

    print(f"\nStandalone RGB Stream Accuracy:  {acc_rgb * 100:.2f}%")
    print(f"Standalone Flow Stream Accuracy: {acc_flow * 100:.2f}%")

    # 3. Clean Alpha Sweep Grid Search
    alphas = np.round(np.arange(0.0, 1.0 + alpha_step / 2, alpha_step), 2)
    best_acc = -1.0
    best_alpha = 0.5
    best_fused_outputs = None

    print("\n" + "="*62)
    print("ALPHA SWEEP GRID SEARCH (RGB vs. Flow Fusion Weight)")
    print("="*62)
    print(f"{'Alpha (RGB)':<12} | {'1 - Alpha (Flow)':<18} | {'Test Accuracy':<15}")
    print("-" * 62)

    for alpha in alphas:
        # Vectorized weighted logit combination
        fused = alpha * video_logits_rgb + (1.0 - alpha) * video_logits_flow
        preds = torch.argmax(fused, dim=1)
        corrects = torch.sum(preds == video_targets).item()
        acc = corrects / len(video_targets)

        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_alpha = float(alpha)
            best_fused_outputs = fused
            marker = " <-- BEST"

        print(f"{alpha:<12.2f} | {1.0 - alpha:<18.2f} | {acc * 100:>6.2f}% {marker}")

    print("="*62)
    print(f"Optimal Alpha Selected: {best_alpha:.2f} (RGB: {best_alpha*100:.0f}%, Flow: {(1-best_alpha)*100:.0f}%)")
    print(f"Best Dual-Stream Test Accuracy: {best_acc * 100:.4f}%\n")

    return video_targets.numpy(), best_fused_outputs.numpy(), best_acc, best_alpha


def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}


from sklearn.metrics import classification_report, confusion_matrix

def get_test_report_logits(targets, outputs, target_names):
    """
    Generates a text classification report. Accepts raw logits or predicted class indices.
    """
    preds = np.argmax(outputs, axis=1) if outputs.ndim > 1 else outputs
    return classification_report(targets, preds, target_names=target_names, digits=4)

def get_confusion_matrix_logits(targets, outputs, labels_dict, all_cats):
    """
    Computes confusion matrix array from raw logits or predicted class indices.
    """
    preds = np.argmax(outputs, axis=1) if outputs.ndim > 1 else outputs
    return confusion_matrix(targets, preds)