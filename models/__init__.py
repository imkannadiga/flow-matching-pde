"""Model package; applies tensorly-torch / PyTorch empty-tensor compatibility before submodules load."""

from models.tltorch_compat import apply_spectral_weight_empty_compat

apply_spectral_weight_empty_compat()
