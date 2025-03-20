import pytorch_lightning as pl
import torch
from beartype import beartype
from jaxtyping import Num, Float
from torch import nn


@beartype
class BaseClassifier(pl.LightningModule):
    """A model that extracts the desired label of a given
    embedded lithology description. The model assumes that the
    description has been embedded using a suitable transformer model.

    Args:
        input_size (int): The size of the input embeddings.
        num_classes (int): The number of classes to predict.
        learning_rate (float): The learning rate for the optimizer.
    """

    def __init__(self, input_size: int, num_classes: int, learning_rate: float):
        super().__init__()

        self.save_hyperparameters()

        # Define a classifier on top of the embeddings
        self.layer_1 = nn.Linear(input_size, 50)
        self.layer_2 = nn.Linear(50, num_classes)
        self.classifier = nn.Sequential(self.layer_1, nn.ReLU(), self.layer_2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, inputs: Float[torch.Tensor, "N D"]) -> Float[torch.Tensor, "N C"]:
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): The input embeddings of shape (N, D).
                where N is the number of samples and D is the embedding size.

        Returns:
            torch.Tensor: The logits of shape (N, C), where C is the number of classes
        """
        logits = self.classifier(inputs)
        return logits

    def training_step(self, batch: list[Num[torch.Tensor, "N ..."]], batch_idx: int):
        """The training step of the model.

        Args:
            batch (list[torch.Tensor]): The input embeddings and labels.
            batch_idx (int): The index of the batch.
        """
        encoded_inputs, labels = batch
        logits = self(encoded_inputs)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[Num[torch.Tensor, "N ..."]], batch_idx: int):
        """The validation step of the model.

        Args:
            batch (list[torch.Tensor]): The input embeddings and labels.
            batch_idx (int): The index of the batch.
        """
        encoded_inputs, labels = batch
        logits = self(encoded_inputs)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: list[Num[torch.Tensor, "N ..."]], batch_idx: int):
        """The test step of the model.

        Args:
            batch (list[torch.Tensor]): The input embeddings and labels.
            batch_idx (int): The index of the batch.
        """
        encoded_inputs, labels = batch
        logits = self(encoded_inputs)
        loss = self.loss_fn(logits, labels)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
