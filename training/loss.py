"""
Loss functions for training neural operators.
"""

class MSE:
    """
    Mean Squared Error (MSE) loss function.
    """
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        """
        Compute the MSE loss between predictions and targets.
        
        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.
        
        Returns:
            torch.Tensor: Computed MSE loss.
        """
        # Sum over all elements so Trainer normalization (divide by n_samples) matches mean loss.
        return ((predictions - targets) ** 2).sum()