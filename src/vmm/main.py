import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from vmm.configs.config import LithologyConfig, ModelConfig
from vmm.data.dataloader import TextEmbeddingDataModule
from vmm.data.preprocess import LithoData
from vmm.models.independent import BaseClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_config: LithologyConfig) -> LithoData:
    """
    Load and preprocess the lithology data.

    Args:
        data_config (LithologyConfig): Configuration for the lithology dataset.

    Returns:
        LithoData: Preprocessed lithology data.
    """
    lithology_data = LithoData(
        file_path=Path(data_config.file_path),
        description_col=data_config.description_col,
        primary_texture_col=data_config.primary_texture_col,
        secondary_texture_col=data_config.secondary_texture_col,
        primary_admixture_col=data_config.primary_admixture_col,
        secondary_admixture_col=data_config.secondary_admixture_col,
        color_col=data_config.color_col,
        combine_labels=data_config.combine_labels,
    )
    lithology_data.preprocess_data()

    return lithology_data


def get_dataloaders(data: LithoData, model_config: ModelConfig) -> TextEmbeddingDataModule:
    """Get the dataloaders for the model.

    Args:
        data (LithoData): Preprocessed lithology data.
        model_config (ModelConfig): Configuration for the model.

    Returns:
        TextEmbeddingDataModule: Data module containing dataloaders for the model
            with embeddings from the transformer model.
    """
    data_module = TextEmbeddingDataModule(
        texts=data.df[model_config.description_col].to_list(),
        labels=data.df[model_config.target_col].to_list(),
        model_name=model_config.model_name,
        save_dir=model_config.precomputed_embeddings_dir,
        batch_size=model_config.batch_size,
        val_split=model_config.val_split,
        test_split=model_config.test_split,
    )

    if not model_config.use_precomputed_embeddings:
        logger.info("Computing embeddings for the data, this may take a while...")
        data_module.setup()
    else:
        logger.info("Loading precomputed embeddings...")
        for f in ["train", "val", "test"]:
            data_module.load_embedding_dataset(
                csv_path=f"{model_config.precomputed_embeddings_dir}/{f}.csv", dataset_type=f
            )
        data_module.load_label_mapper(
            f"{model_config.precomputed_embeddings_dir}/label_mappings.json"
        )

    return data_module


def train_model(data_module: TextEmbeddingDataModule, model_config: ModelConfig) -> BaseClassifier:
    """
    Train the model.

    Args:
        data_module (TextEmbeddingDataModule): Data module containing dataloaders for the model
            with embeddings from the transformer model.
        model_config (ModelConfig): Configuration for the model.

    Returns:
        BaseClassifier: Trained model.
    """
    model = BaseClassifier(
        input_size=model_config.input_size,
        num_classes=len(data_module.label2id),
        learning_rate=model_config.learning_rate,
        dropout=model_config.dropout,
    )
    train_dataloader, val_dataloader, _ = data_module.get_dataloaders()

    trainer = pl.Trainer(max_epochs=model_config.num_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)

    return model


def load_model(model_path: str) -> BaseClassifier:
    """
    Load a trained model.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        BaseClassifier: Trained model.
    """
    logger.info(f"Loading model from {model_path}")
    model = BaseClassifier.load_from_checkpoint(model_path)
    return model


def evaluate_model(model: BaseClassifier, data_module: TextEmbeddingDataModule):
    """
    Evaluate the model.

    Args:
        model (BaseClassifier): Trained model.
        data_module (TextEmbeddingDataModule): Data module containing dataloaders for the model
            with embeddings from the transformer model.
    """
    _, _, test_dataloader = data_module.get_dataloaders()
    actual, predicted, predicted_scores = model.get_predictions(test_dataloader)
    print("Accuracy:", np.round(accuracy_score(actual, predicted), 4))

    print(
        "Top-3 Accuracy:",
        np.round(top_k_accuracy_score(actual, predicted_scores, k=3), 4),
    )


if __name__ == "__main__":
    lithology_config = LithologyConfig.from_yaml("vmm/configs/data_config.yaml")
    model_config = ModelConfig.from_yaml("vmm/configs/model_config.yaml")

    lithology_data = load_and_preprocess_data(lithology_config)

    data_module = get_dataloaders(lithology_data, model_config)

    if model_config.load_model:
        model = load_model(model_config.load_model)
    else:
        model = train_model(data_module, model_config)

    evaluate_model(model, data_module)
