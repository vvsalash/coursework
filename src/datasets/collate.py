import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    result_batch["spec"] = pad_sequence(
        [sample["spectrogram"].squeeze(0).permute(1, 0) for sample in dataset_items],
        batch_first=True,
    ).permute(0, 2, 1)

    result_batch["length"] = torch.tensor(
        [sample["spectrogram"].size(1) for sample in dataset_items]
    )

    result_batch["audio"] = torch.stack([sample["audio"] for sample in dataset_items])

    return result_batch
