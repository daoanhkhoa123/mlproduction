from src.dataloaders.hallu_dataloader import HuggingFaceDataFrame
from transformers import TextClassificationPipeline, Trainer
from torch  import Tensor
from typing import Any, Iterable, Callable

class HalluSentenceClassifier(TextClassificationPipeline):
    def set_concatfn(self, concat_fn:Callable):
        self.concat_fn = concat_fn

    def preprocess(self, inputs:Iterable[str], **tokenizer_kwargs) -> dict[str, list[Tensor] | Tensor | Any]:
        input = self.concat_fn(inputs)
        return super().preprocess(input, **tokenizer_kwargs)
    

def build_pipeline(trainer:Trainer, hf_ds:HuggingFaceDataFrame, **kwargs) -> HalluSentenceClassifier:
    pipeline =  HalluSentenceClassifier(
        task="text-classification", 
        model=trainer.model, 
        tokenizer=trainer.tokenizer,
        **kwargs
    )
    pipeline.set_concatfn(hf_ds.concat_fn) # type: ignore
    return pipeline