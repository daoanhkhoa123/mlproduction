from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def train_test_split_df(df: pd.DataFrame, target_col: str, label_encoder:Optional[LabelEncoder], 
                        test_size: float = 0.2, random_state: int = 42, stratify: bool = True,
                        verbose=False):
    """
    Split a DataFrame into train and test sets.

    Parameters:
        df (pd.DataFrame): The dataset.
        target_col (str): Name of the target column.
        test_size (float): Proportion of test set (default 0.2).
        random_state (int): Random seed for reproducibility.
        stratify (bool): Whether to stratify by target column.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(target_col, axis=1)
    y = label_encoder.transform(df[target_col]) if label_encoder else df[target_col]

    stratify_arg = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )

    if verbose:
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        print("\nSample rows from X_train:")
        print(X_train.head(2))   
        print("\nSample values from y_train:", y_train[:2])

    return X_train, X_test, y_train, y_test

def save_label_encoder(le:LabelEncoder, filepath:str):
    pd.Series(le.classes_).to_csv(filepath, index=False)
    
def load_label_enocder(filepath:str):
    classes = pd.read_csv(filepath).iloc[:, 0].tolist()
    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=object)
    return le