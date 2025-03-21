import numpy as np
import pytorch_lightning as pl
import torch
from beartype import beartype
from jaxtyping import Float, Num
from torch import nn
from torch.utils.data import DataLoader


@beartype
class BaseClassifier(pl.LightningModule):
    """A model that extracts the desired label of a given
    embedded lithology description. The model assumes that the
    description has been embedded using a suitable transformer model.

    Args:
        input_size (int): The size of the input embeddings.
        num_classes (int): The number of classes to predict.
        learning_rate (float): The learning rate for the optimizer.
        dropout (float): The dropout rate for the model.
    """

    def __init__(self, input_size: int, num_classes: int, learning_rate: float, dropout: float):
        super().__init__()

        self.save_hyperparameters()

        # Define a classifier on top of the embeddings
        self.layer_1 = nn.Linear(input_size, 50)
        self.layer_2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(self.layer_1, nn.ReLU(), self.dropout, self.layer_2)
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

    @torch.no_grad()
    def get_predictions(
        self, dataloader: DataLoader
    ) -> tuple[Num[np.ndarray, "N"], Num[np.ndarray, "N"], Float[np.ndarray, "N C"]]:
        """Evaluate the model on the given dataloader.

        Args:
            dataloader (DataLoader): The dataloader to evaluate the model on.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The actual labels, predicted labels, and predicted scores.
        """
        self.eval()
        actual = []
        predicted = []
        predicted_score = []
        for batch in dataloader:
            x, y = batch
            logits = self(x)
            predicted_score.append(logits.cpu().detach().numpy())
            preds = torch.argmax(logits, dim=1)
            actual.append(y.cpu().detach().numpy())
            predicted.append(preds.cpu().detach().numpy())

        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        predicted_score = np.concatenate(predicted_score)

        return actual, predicted, predicted_score
