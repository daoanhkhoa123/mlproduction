from typing import Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_label_encoder(df: pd.DataFrame) -> tuple[LabelEncoder, Dict[str, int], Dict[int, str]]:
    le = LabelEncoder()
    le.fit(df["label"])

    label2id: Dict[str, int] = {
        label: int(idx)
        for idx, label in enumerate(le.classes_)
    }

    id2label: Dict[int, str] = {
        idx: label
        for label, idx in label2id.items()
    }

    return le, label2id, id2label
