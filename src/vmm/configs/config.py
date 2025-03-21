import typing as tp
from dataclasses import dataclass

import yaml
from beartype import beartype


@beartype
@dataclass
class LithologyConfig:
    """
    Dataclass to store the configuration for the lithology dataset.

    Args:
        file_path (str): Path to the file containing the lithology data.
        description_col (str): Column name containing the description of the lithology.
        primary_texture_col (str): Column name containing the primary texture of the lithology.
        secondary_texture_col (str): Column name containing the secondary texture of the lithology.
        primary_admixture_col (str): Column name containing the primary admixture of the lithology.
        secondary_admixture_col (str): Column name containing the secondary admixture of the lithology.
        combine_labels (bool): Whether to combine the primary and secondary texture and admixture labels.
        color_col (str): Column name containing the color of the lithology.
    """

    file_path: str
    description_col: str
    primary_texture_col: str
    secondary_texture_col: str
    primary_admixture_col: str
    secondary_admixture_col: str
    combine_labels: bool
    color_col: str

    @classmethod
    def from_yaml(cls, yaml_path: str) -> tp.Self:
        """Load the lithology configuration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing the lithology configuration.

        Returns:
            LithologyConfig: Lithology configuration
        """
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


@beartype
@dataclass
class ModelConfig:
    """
    Dataclass to store the configuration for the model.

    Args:
        model_name (str): Name of the model to use.
        val_split (float): Fraction of the data to use for validation.
        test_split (float): Fraction of the data to use for testing.
        input_size (int): Size of the input to the model.
        learning_rate (float): Learning rate for the optimizer.
        dropout (float): Dropout rate for the model.
        batch_size (int): Batch size for the data loader.
        num_epochs (int): Number of epochs to train the model.
        precomputed_embeddings_dir (str): Path to dir to store precomputed embeddings.
        use_precomputed_embeddings (bool): Whether to use precomputed embeddings.
        description_col (str): Column name containing the description of the lithology.
        target_col (str): Column name containing the target label.
    """

    model_name: str
    val_split: float
    test_split: float
    input_size: int
    learning_rate: float
    dropout: float
    batch_size: int
    num_epochs: int
    precomputed_embeddings_dir: str
    use_precomputed_embeddings: bool
    description_col: str
    target_col: str

    @classmethod
    def from_yaml(cls, yaml_path: str) -> tp.Self:
        """Load the model configuration from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing the model configuration.

        Returns:
            ModelConfig: Model configuration
        """
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)
