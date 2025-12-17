from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class TextPairDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase,
        label_encoder: LabelEncoder,
        max_length: int = 512) -> None:
        
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["context"],
            row["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            k: v.squeeze(0) for k, v in encoding.items()
        }

        label_id: int = int(self.le.transform([row["label"]])[0]) # type: ignore
        item["labels"] = torch.tensor(label_id, dtype=torch.long)

        return item


class TextPairDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame, val_df: pd.DataFrame, 
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8, max_length: int = 512, num_workers: int = 4) -> None:
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.label_encoder: Optional[LabelEncoder] = None
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.label_encoder = LabelEncoder()
        self.train_df["label_id"] = self.label_encoder.fit_transform(self.train_df["label"])
        self.val_df["label_id"] = self.label_encoder.transform(self.val_df["label"])

        self.train_ds = TextPairDataset(
            self.train_df,
            self.tokenizer,
            self.label_encoder,
            self.max_length,
        )

        self.val_ds = TextPairDataset(
            self.val_df,
            self.tokenizer,
            self.label_encoder,
            self.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
