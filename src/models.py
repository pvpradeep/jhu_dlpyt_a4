"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights, ResNet50_Weights

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """
    LRCN (Long-term Recurrent Convolutional Network) for video classification.
    
    This model uses a ResNet backbone as a 2D CNN to extract spatial features from each video frame.
    An LSTM network is then used to model the temporal dynamics across the sequence of frame features.
    Dropout is applied before the final fully-connected layer that produces class logits.

    Args:
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        dropout_rate (float): Dropout rate applied before the final classification layer.
        n_classes (int): Number of output classes.
        pretrained (bool, optional): If True, uses a ResNet model pretrained on ImageNet. Default is True.
        cnn_model (str, optional): Specifies the ResNet variant to use as the backbone.
                                   Options: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.
                                   Default is 'resnet34'.
    
    Raises:
        ValueError: If the specified cnn_model is not supported.
    """
    def __init__(self, hidden_size, n_layers, dropout_rate, n_classes, pretrained=True, cnn_model='resnet34'):
        super(LRCN, self).__init__()

        # Set up the ResNet backbone as a 2D CNN feature extractor.
        if cnn_model == 'resnet18':
            base_cnn = models.resnet18(pretrained=pretrained)
        elif cnn_model == 'resnet34':
            base_cnn = models.resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        elif cnn_model == 'resnet50':
            base_cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif cnn_model == 'resnet101':
            base_cnn = models.resnet101(pretrained=pretrained)
        elif cnn_model == 'resnet152':
            # Note: This example uses resnet34 for resnet152 option as a placeholder.
            base_cnn = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError('The input CNN backbone is not supported, please choose a valid ResNet variant.')

        # Retrieve the number of features output by the CNN's original fully-connected layer.
        num_features = base_cnn.fc.in_features
        
        # Replace the original fc layer with an identity mapping so that raw features are returned.
        base_cnn.fc = Identity()
        self.base_model = base_cnn

        # Define the LSTM to process the sequence of frame features.
        self.rnn = nn.LSTM(num_features, hidden_size, n_layers, batch_first=True,
                           bidirectional=True)
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size * 2, n_classes)
    
    def freeze_backbone(self):
        """Freeze all CNN backbone parameters to prevent overfitting."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze CNN backbone parameters for fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward_orig(self, x):
        """
        Forward pass for the LRCN model.
        
        The input tensor x is expected to have the shape:
            (batch_size, time_steps, channels, height, width)
        
        For each time step (frame), the CNN backbone extracts features. These features are then
        passed through the LSTM sequentially. The output from the last time step is then passed
        through dropout and the final fully-connected layer to produce the class logits.

        Args:
            x (Tensor): Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
            Tensor: Output logits for each sample in the batch with shape (batch_size, n_classes).
        """
        bs, ts, c, h, w = x.shape  # batch_size, time_steps, channels, height, width
        
        # Process the first frame separately to initialize the LSTM hidden and cell states.
        idx = 0
        y = self.base_model(x[:, idx])
        _, (hn, cn) = self.rnn(y.unsqueeze(1))

        # Iterate over the remaining frames, feeding each frame's features into the LSTM.
        for idx in range(1, ts):
            y = self.base_model(x[:, idx])
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))

        # Apply dropout to the output of the final time step.
        out = self.dropout(out[:, -1])

        # Pass the final output through the fully-connected layer to get class logits.
        out = self.fc(out)
        return out

    def forward(self, x):
        """
            # ALL frames through CNN in one pass, ALL timesteps through LSTM in one pass
            to eliminate the for loop and speed up the forward pass.
            c_in = x.view(bs * ts, c, h, w)   # (4*16, 3, 224, 224) = (64, 3, 224, 224)
            c_out = self.base_model(c_in)      # CNN forward pass × 1
            r_in = c_out.view(bs, ts, -1)      # (4, 16, 512)
            out, (hn, cn) = self.rnn(r_in)     # LSTM forward pass × 1

            forward_orig:  CNN processes (4, 3, 224, 224)   per step = 4 images
            forward:       CNN processes (64, 3, 224, 224)  at once  = 64 images

        """
        bs, ts, c, h, w = x.shape
        
        # Fold batch and sequence dimensions to process all frames through CNN in parallel
        c_in = x.view(bs * ts, c, h, w)
        c_out = self.base_model(c_in)  # shape: (bs * ts, num_features)
        
        # Reshape back to sequence format: (batch_size, time_steps, num_features)
        r_in = c_out.view(bs, ts, -1)
        
        # Pass entire sequence through LSTM at once
        out, (hn, cn) = self.rnn(r_in)
        
        # Take the output from the final frame timestep
        # out = self.dropout(out[:, -1, :])

        # Taking the mean representation across all timesteps
        # captures video-wide context far better than relying
        # solely on the final frame's hidden state
        # Mean pool across all timesteps instead of last-only
        out = self.dropout(out.mean(dim=1))

        out = self.fc(out)
        return out