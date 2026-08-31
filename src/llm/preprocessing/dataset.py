import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataset(Dataset):
    """Creates input-target pairs for next-token prediction."""

    def __init__(
        self,
        text: str,
        tokenizer,
        max_length: int,
        stride: int,
    ):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(
            0,
            len(token_ids) - max_length,
            stride,
        ):
            input_chunk = token_ids[
                i : i + max_length
            ]

            target_chunk = token_ids[
                i + 1 : i + max_length + 1
            ]

            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(
                self.input_ids[idx],
                dtype=torch.long,
            ),
            torch.tensor(
                self.target_ids[idx],
                dtype=torch.long,
            ),
        )


def create_data_loader(
    text: str,
    batch_size: int = 5,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    """Create a DataLoader for GPT training."""

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader