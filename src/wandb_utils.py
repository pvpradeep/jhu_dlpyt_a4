"""Weights & Biases integration for UCF50 action recognition."""

import os

import wandb
from dotenv import load_dotenv


def init_wandb(args, run_name=None):
    """Initialize Weights & Biases run.

    Args:
        args: argparse.Namespace with run configuration.
        run_name: Optional custom run name.

    Returns:
        wandb.Run or None if init fails.
    """
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        print("WARNING: WANDB_API_KEY not found in .env — skipping W&B logging")
        return None

    wandb.login(key=api_key)

    run = wandb.init(
        project="ucf50-action-recognition",
        job_type="training" if args.mode == "train" else "evaluation",
        name=run_name,
        config=vars(args),
    )
    return run


def log_epoch(epoch, train_loss, val_loss, val_accuracy, lr):
    """Log epoch metrics to W&B.

    Args:
        epoch: Current epoch number.
        train_loss: Training loss for the epoch.
        val_loss: Validation loss for the epoch.
        val_accuracy: Validation accuracy percentage.
        lr: Current learning rate.
    """
    if wandb.run:
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
            "learning_rate": lr,
        })


def log_test(test_loss, test_accuracy):
    """Log test metrics to W&B.

    Args:
        test_loss: Test loss.
        test_accuracy: Test accuracy percentage.
    """
    if wandb.run:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
        })
        wandb.summary.update({
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
        })


def finish_wandb(best_val_accuracy=None, best_epoch=None):
    """Finalize W&B run with summary metrics.

    Args:
        best_val_accuracy: Best validation accuracy achieved.
        best_epoch: Epoch where best accuracy was achieved.
    """
    if wandb.run:
        if best_val_accuracy is not None:
            wandb.summary.update({
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
            })
        wandb.finish()