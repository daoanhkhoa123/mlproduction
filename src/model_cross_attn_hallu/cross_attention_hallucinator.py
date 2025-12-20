       
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "vinai/phobert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_)  # change to your number of classes
)
