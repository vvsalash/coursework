import torch

from src.metrics.base_metric import BaseMetric

# macs, flops, rtf, stoi, pesq, csig, cbak, covl,

# https://github.com/SamsungLabs/hifi_plusplus/blob/main/metric_denoising.py#L13


class SISNRMetric(BaseMetric):
    """
    Signal-to-Noise Ratio (SI-SNR) metric for audio quality evaluation.
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        super().__init__(name=name, *args, **kwargs)

    def _compute_si_snr(self, s_pred: torch.Tensor, s_true: torch.Tensor):
        """
        Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

        Args:
            s_pred (Tensor): Predicted audio signal (B, T).
            s_true (Tensor): Ground-truth audio signal (B, T).

        Returns:
            si_snr (Tensor): Scale-Invariant SNR for each sample in the batch.
        """

        s_pred_mean = s_pred.mean(dim=-1, keepdim=True)
        s_true_mean = s_true.mean(dim=-1, keepdim=True)

        s_pred = s_pred - s_pred_mean
        s_true = s_true - s_true_mean

        dot_product = torch.sum(s_true * s_pred, dim=-1, keepdim=True)
        s_true_energy = torch.sum(s_true**2, dim=-1, keepdim=True)

        s_proj = dot_product / (s_true_energy + 1e-8) * s_true

        e_noise = s_pred - s_proj

        si_snr = 10 * torch.log10(
            torch.sum(s_proj**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + 1e-8)
        )

        return si_snr

    def __call__(self, s_pred: torch.Tensor, s_true: torch.Tensor, **kwargs):
        """
        Args:
            s_pred (Tensor): Predicted audio signal (B, T).
            s_true (Tensor): Ground-truth audio signal (B, T).

        Returns:
            metric (float): Mean SI-SNR across the batch.
        """

        si_snr = self._compute_si_snr(s_pred, s_true)
        return si_snr.mean().item()
