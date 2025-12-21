from src.dataloaders.hallu_dataloader import HuggingFaceDataFrame
from transformers import TextClassificationPipeline, Trainer, AutoTokenizer, AutoModelForSequenceClassification, AddedToken
from sklearn.preprocessing import LabelEncoder
from torch  import Tensor
from typing import Any, Iterable, Callable, Tuple

class HalluSentenceClassifier(TextClassificationPipeline):
    def set_concatfn(self, concat_fn:Callable):
        self.concat_fn = concat_fn

    def preprocess(self, inputs:Iterable[str], **tokenizer_kwargs) -> dict[str, list[Tensor] | Tensor | Any]:
        input = self.concat_fn(*inputs)
        return super().preprocess(input, **tokenizer_kwargs)
    
def build_tokenizer_model(model_name:str, le:LabelEncoder, hf_ds:HuggingFaceDataFrame,
                          tokenizer_kwargs=dict(), model_kwargs=dict()) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    id2label = {i: label for i, label in enumerate(le.classes_)}
    label2id = {label: i for i, label in enumerate(le.classes_)}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le.classes_),
        id2label=id2label, label2id=label2id,
        **model_kwargs)


    special_tokens = {"additional_special_tokens": [AddedToken(f"[{ic}]", single_word=True, normalized=False) for ic in hf_ds.input_columns]}
    tokenizer.add_special_tokens(special_tokens)
    print("Added token:", special_tokens["additional_special_tokens"])
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def build_pipeline(trainer:Trainer, hf_ds:HuggingFaceDataFrame, **kwargs) -> HalluSentenceClassifier:
    pipeline =  HalluSentenceClassifier(
        task="text-classification", 
        model=trainer.model, 
        tokenizer=trainer.tokenizer,
        **kwargs
    )
    pipeline.set_concatfn(hf_ds.concat_fn) # type: ignore
    return pipeline 