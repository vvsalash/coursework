import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class LSDMetric(BaseMetric):
    """
    Log-Spectral Distance (LSD) metric for audio quality evaluation.
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): Metric name to use in logger and writer.
        """
        super().__init__(name=name, *args, **kwargs)

    def __call__(self, s_pred: torch.Tensor, s_true: torch.Tensor, **kwargs):
        """
        Args:
            s_pred (Tensor): Predicted mel-spectrogram (B, F, T).
            s_true (Tensor): Ground-truth mel-spectrogram (B, F, T).

        Returns:
            metric (float): Mean LSD across the batch.
        """
        if s_pred.size() != s_true.size():
            raise ValueError(
                "Predicted and ground-truth mel-spectrograms must have the same shape."
            )

        eps = 1e-8
        log_s_pred = torch.log(s_pred + eps)
        log_s_true = torch.log(s_true + eps)

        lsd = torch.sqrt(torch.mean((log_s_pred - log_s_true) ** 2), dim=(1, 2))

        return lsd.mean().item()
