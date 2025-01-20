import torch

from src.metrics.base_metric import BaseMetric


class MOSMetric(BaseMetric):
    """
    Mean Opinion Score (MOS) metric using a pretrained MOS prediction model.
    """

    def __init__(self, model, device, name=None, *args, **kwargs):
        """
        Args:
            model (Callable): Pretrained MOS prediction model.
            device (str): Device for metric computation (e.g., 'cpu' or 'cuda').
        """
        super().__init__(name, *args, **kwargs)
        self.model = model.to(device)
        self.device = device

    def __call__(self, audio: torch.Tensor, **kwargs):
        """
        Args:
            audio (Tensor): Input audio signal (B, T).

        Returns:
            metric (float): Predicted MOS value.
        """
        audio = audio.to(self.device)
        with torch.no_grad():
            mos = self.model(audio)
        return mos.mean().item()
