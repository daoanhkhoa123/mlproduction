# Load model directly
from typing import Any
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")

import torch

text = "(attended): KhoaDACE190399 had attended this activity / Đào Anh Khoa đã tham gia hoạt động này (absent): KhoaDACE190399 had NOT attended this activity / Đào Anh Khoa đã vắng mặt buổi này (-): no data was given / chưa có dữ liệu"

tokens = tokenizer(text, return_tensors="pt")
print(tokens)

with torch.no_grad():
    output= model(**tokens)

output["logits"].shape

import lightning as lt

class AttentionHallucination(lt.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


