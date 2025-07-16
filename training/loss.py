# TODO

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
        return ((predictions - targets) ** 2).mean()