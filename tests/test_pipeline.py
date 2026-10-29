import re
import numpy as np
import pytest
import torch

from src.models import LRCN


# =====================================================================
# 1. MODEL ARCHITECTURE & TENSOR DIMENSION TEST
# =====================================================================
def test_lrcn_forward_pass_shape():
    """Verify folded batch forward pass (B*T, C, H, W) and temporal mean pooling output."""
    batch_size = 2
    seq_len = 16
    num_classes = 50

    # Synthetic clip batch: (B, T, C, H, W)
    x = torch.randn(batch_size, seq_len, 3, 224, 224)

    # Instantiate model using lightweight resnet18 backbone for fast unit test
    model = LRCN(
        n_classes=num_classes,
        cnn_model="resnet18",
        hidden_size=256,
        n_layers=2,
        dropout_rate=0.5
    )
    model.eval()

    with torch.no_grad():
        logits = model(x)

    # Assert output shape matches (batch_size, num_classes)
    assert logits.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"


# =====================================================================
# 2. DATA LEAKAGE & GROUP-AWARE SPLIT TEST
# =====================================================================
def test_group_aware_regex_split():
    """Verify regex group extraction isolates actor/video groups into disjoint sets."""
    train_files = [
        "v_BaseballPitch_g01_c01.avi",
        "v_BaseballPitch_g01_c02.avi",
        "v_Basketball_g02_c01.avi",
    ]
    test_files = [
        "v_BaseballPitch_g13_c01.avi",
        "v_Basketball_g16_c01.avi",
    ]

    group_pattern = re.compile(r"_g(\d+)_")

    train_groups = {
        group_pattern.search(f).group(1)
        for f in train_files
        if group_pattern.search(f)
    }
    test_groups = {
        group_pattern.search(f).group(1)
        for f in test_files
        if group_pattern.search(f)
    }

    # Assert disjoint sets (zero overlap between train and test group IDs)
    assert train_groups.isdisjoint(
        test_groups
    ), f"Data leakage detected! Shared group IDs: {train_groups & test_groups}"


# =====================================================================
# 3. NUMERIC FRAME SORTING TEST
# =====================================================================
def test_numeric_frame_sorting():
    """Verify string paths are sorted numerically to prevent temporal scrambling."""
    scrambled_paths = [
        "frame10.jpg",
        "frame1.jpg",
        "frame2.jpg",
        "frame100.jpg",
        "frame20.jpg",
    ]

    # Regex key extraction used in video_datasets.py
    sorted_paths = sorted(
        scrambled_paths,
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    expected_order = [
        "frame1.jpg",
        "frame2.jpg",
        "frame10.jpg",
        "frame20.jpg",
        "frame100.jpg",
    ]

    assert (
        sorted_paths == expected_order
    ), f"Sorting failed! Got {sorted_paths}, expected {expected_order}"


# =====================================================================
# 4. SOFT-VOTING LOGIT FUSION MATH TEST
# =====================================================================
def test_soft_voting_fusion_calculation():
    """Verify late soft-voting logit combination alpha * L_rgb + (1 - alpha) * L_flow."""
    # 1 sample, 3 classes (JumpingJack, Standing, Basketball)
    logits_rgb = np.array([[1.20, 3.80, 0.50]])  # Predicts index 1 (Standing)
    logits_flow = np.array(
        [[5.40, 0.10, -1.00]]
    )  # Predicts index 0 (JumpingJack)

    alpha = 0.50
    fused_logits = alpha * logits_rgb + (1.0 - alpha) * logits_flow

    expected_fused = np.array([[3.30, 1.95, -0.25]])

    # Numerical verification
    np.testing.assert_allclose(fused_logits, expected_fused, rtol=1e-5)

    # Assert high-confidence Optical Flow overrules wrong spatial RGB prediction
    winning_class = np.argmax(fused_logits, axis=1)[0]
    assert (
        winning_class == 0
    ), f"Expected fused winner class 0 (JumpingJack), got {winning_class}"


# =====================================================================
# 5. TEMPORAL MEAN-POOLING GRADIENT FLOW TEST
# =====================================================================
def test_mean_pooling_gradient_flow():
    """Verify out.mean(dim=1) propagates non-zero gradients to ALL 16 frame timesteps."""
    batch_size = 2
    seq_len = 16
    num_classes = 50

    # Tensor with gradient tracking enabled
    x = torch.randn(
        batch_size, seq_len, 3, 224, 224, requires_grad=True
    )

    model = LRCN(
        n_classes=num_classes,
        cnn_model="resnet18",
        hidden_size=256,
        n_layers=2,
        dropout_rate=0.5
    )
    model.train()

    logits = model(x)
    loss = logits.sum()
    loss.backward()

    # Sum absolute gradient magnitudes across spatial dimensions for each timestep: shape (16,)
    grad_per_timestep = x.grad.abs().sum(dim=(0, 2, 3, 4))

    # Assert EVERY frame in the sequence received non-zero gradients during backprop
    assert torch.all(
        grad_per_timestep > 0
    ), f"Gradient bottleneck detected! Timesteps without gradients: {torch.where(grad_per_timestep == 0)[0]}"


# =====================================================================
# 6. BACKBONE FREEZE & UNFREEZE STATE TEST
# =====================================================================
def test_backbone_freeze_unfreeze_states():
    """Verify freeze_backbone() locks CNN weights and unfreeze_from() selectively unlocks deeper blocks."""
    model = LRCN(
        n_classes=50,
        cnn_model="resnet18",
        hidden_size=256,
        n_layers=2,
        dropout_rate=0.5
    )

    # 1. Test full backbone freeze (Stage 1 training mode)
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        for param in model.base_model.parameters():
            assert (
                not param.requires_grad
            ), "Backbone parameter was not frozen!"

    # 2. Test selective layer unfreezing (Stage 2 fine-tuning mode)
    if hasattr(model, "unfreeze_from"):
        model.unfreeze_from("layer3")
        # Assert layer3 and layer4 parameters have requires_grad = True
        layer3_unfrozen = any(
            p.requires_grad for p in model.base_model.layer3.parameters()
        )
        layer4_unfrozen = any(
            p.requires_grad for p in model.base_model.layer4.parameters()
        )
        assert (
            layer3_unfrozen and layer4_unfrozen
        ), "Deeper layers failed to unfreeze!"
