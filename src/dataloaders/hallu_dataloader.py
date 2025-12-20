from typing import Dict, Optional, Union, Any, Iterable, Callable

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

#NOTE: will be installed in cloud
from datasets import Dataset # type: ignore


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
        tokenizer: PreTrainedTokenizerBase, label_encoder:LabelEncoder,
        batch_size: int = 8, max_length: int = 512, num_workers: int = 4) -> None:
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.le=label_encoder
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_df["label_id"] = self.le.transform(self.train_df["label"])
        self.val_df["label_id"] = self.le.transform(self.val_df["label"])

        self.train_ds = TextPairDataset(
            self.train_df,
            self.tokenizer,
            self.le,
            self.max_length,
        )

        self.val_ds = TextPairDataset(
            self.val_df,
            self.tokenizer,
            self.le,
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


class HuggingFaceDataFrame:
    def __init__(self,*, df = None, input_columns:Iterable[str], concat_fn=None, n_inputs=None) -> None:
        if df is None:
            raise RuntimeError(
                "Do not call HuggingFaceDataFrame() directly. "
                "Use HuggingFaceDataFrame.from_df(...) instead."
            )
        
        self.df = df
        self.concat_fn = concat_fn
        self.dataset = Dataset.from_pandas(df, preserve_index=False)
        self.input_columns = input_columns
        self.n_inputs = n_inputs
    
    @classmethod
    def from_df(cls, df:pd.DataFrame, concat_cols:Iterable[str], target_col:str, le:Optional[LabelEncoder]=None):
        def concat_fn(*args:str):
            assert len(args) == len(concat_cols), f"Number of inputs has to be exactly same as dataset inititalization. Got{len(args)} expected {len(concat_cols)}" # type: ignore
            return "[SEP] ".join(f"[{col.upper()}] {val}" for col, val in zip(concat_cols, args))

        ds_df = pd.DataFrame()
        ds_df["text"] = df[list(concat_cols)].agg(lambda row: concat_fn(*row), axis=1)
        ds_df["label"] = le.transform(df[target_col]) if le is not None else df[target_col]
        
        return cls(df=ds_df, input_columns=concat_cols, concat_fn=concat_fn, n_inputs = len(concat_cols)) # type: ignore

    def train_test_split(self, *args, **kwargs):
        dataset =  self.dataset.train_test_split(*args, **kwargs)
        return dataset["train"], dataset["test"] # type: ignore

    @staticmethod
    def tokenize(dataset: Dataset, tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = "max_length", truncation: bool = True,
        max_length: Optional[int] = None, batched: bool = True, **tokenizer_kwargs: Any) -> Dataset:
        tokenizer_fn = lambda bacth : tokenizer(bacth["text"], padding=padding, 
                                                truncation=truncation, max_length=max_length,
                                                  **tokenizer_kwargs)
        dataset = dataset.map(tokenizer_fn, batched=batched)
        dataset.remove_columns(["text"])
        dataset.set_format("torch")
        return dataset