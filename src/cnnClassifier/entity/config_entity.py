from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_image_size: list
    params_learning_rate: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    Cover_train: Path
    Med_train: Path
    Cover_val:Path
    Med_val:Path
    trained_model_path: Path
    base_model_path: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    Med_test: Path
    Cover_test:Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
