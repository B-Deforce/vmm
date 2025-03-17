from typing import Dict, Tuple

import pandas as pd
import torch
from beartype import beartype
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@beartype
class LithoDataLoader:
    """
    A wrapper for splitting a dataset into stratified train, validation, and test sets,
    and converting them into PyTorch DataLoaders.

    Args:
        df (pd.DataFrame): The input dataset.
        y_col (str): Target column name.
        text_col (str): Text column name.
        tokenizer_name (str, optional): Name of the HuggingFace tokenizer to use. Defaults to None.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str,
        text_col: str,
        tokenizer,
        test_size: float = 0.2,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.df = df
        self.y_col = y_col
        self.text_col = text_col
        self.batch_size = batch_size
        self.random_state = random_state
        self.test_size = test_size
        self.tokenizer = tokenizer
        # Create label mappings
        self.label2id = self._get_label2id(df[y_col])
        self.id2label = {v: k for k, v in self.label2id.items()}  # Reverse mapping

        self._split_data()
        self._create_dataloaders()

    @staticmethod
    def _get_label2id(df_target: pd.Series) -> Dict[str, int]:
        """Creates a mapping from categorical labels to integer IDs.

        Args:
            df_target (pd.Series): Series containing target labels.

        Returns:
            Dict[str, int]: Mapping from label strings to numeric IDs.
        """
        unique_labels = df_target.unique()
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _tokenize_text(self, text_series: pd.Series) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes a given text column.

        Args:
            text_col (pd.Series): Series containing feature columns.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors containing tokenized input_ids and attention_masks.
        """
        # same as Ghorbanfekr at al. (2025)
        tokenized = self.tokenizer(
            text_series.astype(str).tolist(),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return tokenized["input_ids"], tokenized["attention_mask"]

    def _split_data(self):
        """
        Splits the dataset into stratified train, validation (half of test size), and test sets.
        """
        # Tokenize text column
        input_ids, attention_masks = self._tokenize_text(self.df[self.text_col])

        # Convert labels to tensor
        y = torch.tensor(self.df[self.y_col].map(self.label2id).values, dtype=torch.long)

        # Step 1: Train-Test Split (Stratified)
        (
            input_ids_train,
            input_ids_temp,
            attention_masks_train,
            attention_masks_temp,
            y_train,
            y_temp,
        ) = train_test_split(
            input_ids,
            attention_masks,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        # Step 2: Validation-Test Split (Stratified)
        (
            input_ids_val,
            input_ids_test,
            attention_masks_val,
            attention_masks_test,
            y_val,
            y_test,
        ) = train_test_split(
            input_ids_temp,
            attention_masks_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        # Store splits
        self.data_splits = {
            "train": (input_ids_train, attention_masks_train, y_train),
            "val": (input_ids_val, attention_masks_val, y_val),
            "test": (input_ids_test, attention_masks_test, y_test),
        }

    def _create_dataloaders(self):
        """
        Converts the train, validation, and test sets into PyTorch DataLoaders.
        """
        self.dataloaders = {}
        for split in ["train", "val", "test"]:
            input_ids, attention_masks, labels = self.data_splits[split]

            dataset = LithoDataset(input_ids, attention_masks, labels)
            self.dataloaders[split] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True if split == "train" else False
            )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns the train, validation, and test DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The train, validation, and test DataLoaders.
        """
        return self.dataloaders["train"], self.dataloaders["val"], self.dataloaders["test"]


class LithoDataset(Dataset):
    def __init__(self, tokens, masks, labels):
        self.tokens = tokens
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens.squeeze(0)[idx],
            "attention_mask": self.masks.squeeze(0)[idx],
            "labels": self.labels,
        }
