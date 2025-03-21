import gc
import json
import logging
import os
import typing as tp

import pandas as pd
import pytorch_lightning as pl
import torch
from beartype import beartype
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

logger = logging.getLogger(__name__)


@beartype
class TextEmbeddingDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that preprocesses text data using a Hugging Face
    feature extraction pipeline, generates CLS token embeddings, and provides train,
    validation, and test DataLoaders. Additionally, saves data as Pandas DataFrames
    for traditional ML models (e.g., Decision Trees, XGBoost).

    Args:
        texts (list[str]): List of input text strings.
        labels (list[str]): List of text labels corresponding to input texts.
        model_name (str): Pretrained model name. E.g., "hghcomphys/geobertje-base-dutch-uncased".
        batch_size (int, optional): Batch size for PyTorch DataLoaders. Defaults to 32.
        val_split (float, optional): Proportion of data for validation. Defaults to 0.15.
        test_split (float, optional): Proportion of data for testing. Defaults to 0.15.
        seed (int | None, optional): Seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        model_name: str,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        save_dir: str = "./embedding_data/",
        seed: int | None = 42,
    ):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Ensure reproducibility
        pl.seed_everything(seed, workers=True)

        # Placeholder for label mappings
        self.label2id = None
        self.id2label = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def _encode_label(self, label: str) -> int:
        """Encodes string label as an integer.

        Args:
            label (str): Text label.

        Returns:
            int: Encoded label.
        """
        return self.label2id.get(label, -1)

    def setup(self, stage: str | None = None):
        """Splits data into train/val/test, processes embeddings, and saves results.

        Args:
            stage (str | None, optional): Stage of training. Defaults to None.
        """
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = (
            self._split_data()
        )

        self._create_label_mappings(train_labels)

        train_label_ids = self._encode_labels(train_labels)
        val_label_ids = self._encode_labels(val_labels)
        test_label_ids = self._encode_labels(test_labels)

        train_embeddings = self._compute_embeddings(train_texts)
        train_labels_tensor = torch.tensor(train_label_ids)
        self._save_as_dataframe("train.csv", train_embeddings, train_labels_tensor, train_texts)
        self.train_dataset = TensorDataset(train_embeddings, train_labels_tensor)

        # Free memory
        del train_embeddings
        del train_labels_tensor
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU

        logger.info("Train embeddings saved.")

        val_embeddings = self._compute_embeddings(val_texts)
        val_labels_tensor = torch.tensor(val_label_ids)
        self._save_as_dataframe("val.csv", val_embeddings, val_labels_tensor, val_texts)
        self.val_dataset = TensorDataset(val_embeddings, val_labels_tensor)

        # Free memory
        del val_embeddings
        del val_labels_tensor
        gc.collect()
        torch.cuda.empty_cache()  # If using GPU

        logger.info("Validation embeddings saved.")

        test_embeddings = self._compute_embeddings(test_texts)
        test_labels_tensor = torch.tensor(test_label_ids)
        self._save_as_dataframe("test.csv", test_embeddings, test_labels_tensor, test_texts)
        self.test_dataset = TensorDataset(test_embeddings, test_labels_tensor)

        # Free memory
        del test_embeddings
        del test_labels_tensor
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Test embeddings saved.")

    def _split_data(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        """Splits dataset into train, validation, and test sets using stratification."""
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            self.texts,
            self.labels,
            test_size=self.test_split,
            stratify=self.labels,
            random_state=self.seed,
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=self.val_split / (1 - self.test_split),
            stratify=train_labels,
            random_state=self.seed,
        )

        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

    def _create_label_mappings(self, train_labels: list[str]):
        """Creates label-to-ID and ID-to-label mappings using training labels.

        Args:
            train_labels (list[str]): List of training labels.
        """
        unique_train_labels = sorted(set(train_labels))
        self.label2id = {label: i for i, label in enumerate(unique_train_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        with open("label_mappings.json", "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f, indent=4)

    def _encode_labels(self, labels: list[str]) -> list[int]:
        """Converts text labels to numeric IDs, assigning -1 for unknown labels.

        Args:
            labels (list[str]): List of text labels.

        Returns:
            list[int]: List of encoded label IDs.
        """
        return [self.label2id.get(label, -1) for label in labels]

    def _get_tokenizer(self) -> BertTokenizerFast:
        """Returns a Hugging Face tokenizer.

        Returns:
            BertTokenizerFast: A Hugging Face tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_name)

    def _get_model(self) -> BertModel:
        """Returns a Hugging Face model.

        Returns:
            BertModel: A Hugging Face model.
        """
        return AutoModel.from_pretrained(self.model_name)

    def _compute_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Computes CLS token embeddings using the Hugging Face pipeline.

        Args:
            texts (list[str]): List of input text strings.

        Returns:
            torch.Tensor: Tensor of computed embeddings.
        """
        feature_extractor = pipeline(
            "feature-extraction", model=self._get_model(), tokenizer=self._get_tokenizer()
        )
        return torch.tensor(
            [e[0][0] for e in feature_extractor(texts, truncation=True, padding=True)]
        )

    def _save_as_dataframe(
        self, filename: str, embeddings: torch.Tensor, labels: torch.Tensor, texts: list[str]
    ):
        """Saves embeddings and labels as a Pandas DataFrame.

        Args:
            filename (str): Name of the CSV file.
            embeddings (torch.Tensor): Tensor of computed embeddings.
            labels (torch.Tensor): Tensor of encoded label IDs.
            texts (list[str]): List of input text strings.

        """
        df = pd.DataFrame(embeddings.numpy())
        df["label"] = labels.numpy()
        df["text"] = texts
        df.to_csv(filename, index=False)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader for PyTorch training.

        Returns:
            DataLoader: A DataLoader for PyTorch training
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader for PyTorch validation.

        Returns:
            DataLoader: A DataLoader for PyTorch validation
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Returns a DataLoader for PyTorch testing.

        Returns:
            DataLoader: A DataLoader for PyTorch testing
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Returns train, validation, and test DataLoaders.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
        """
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def decode_labels(self, label_ids: list[int]) -> list[str]:
        """Converts label IDs back to text labels (returns 'unknown' if missing).

        Args:
            label_ids (list[int]): List of encoded label IDs.

        Returns:
            list[str]: List of text labels.
        """
        return [self.id2label.get(label_id, "unknown") for label_id in label_ids]

    def _load_torch_dataset(self, csv_path: str) -> TensorDataset:
        """
        Loads embeddings and labels from a CSV file into a PyTorch TensorDataset.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            TensorDataset: A PyTorch dataset containing the embeddings and labels.
        """
        df = pd.read_csv(csv_path)

        embedding_cols = [col for col in df.columns if col not in ["label", "text"]]
        embeddings = torch.tensor(df[embedding_cols].values, dtype=torch.float32)

        labels = torch.tensor(df["label"].values, dtype=torch.long)

        return TensorDataset(embeddings, labels)

    def load_embedding_dataset(
        self, csv_path: str, dataset_type: tp.Literal["train", "val", "test"]
    ):
        """Loads the training dataset into memory from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            dataset_type (str): Type of dataset. One of "train", "val", or "test".
        """
        if dataset_type == "train":
            self.train_dataset = self._load_torch_dataset(csv_path)
        elif dataset_type == "val":
            self.val_dataset = self._load_torch_dataset(csv_path)
        else:
            self.test_dataset = self._load_torch_dataset(csv_path)

    def load_label_mapper(self, json_path: str):
        """Loads label mappings from a JSON file.

        Args:
            json_path (str): Path to the JSON file.
        """
        with open(json_path, "r") as f:
            mappings = json.load(f)
            self.label2id = mappings["label2id"]
            self.id2label = mappings["id2label"]
