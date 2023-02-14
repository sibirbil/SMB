from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from .testoptimizers import TestOptimizer
from .trainingsetups import TrainingSetup


def load_yaml(yaml_file: Path) -> Dict:
    with open(yaml_file, "r") as f:
        dict_yaml = yaml.safe_load(f)

    return dict_yaml


@dataclass
class TrainingSettings:
    setups: List[TrainingSetup]
    seed: Optional[int]

    @staticmethod
    def set_seeds(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def from_yaml(cls, yaml_file: Path):
        settings = load_yaml(yaml_file=yaml_file)

        datasets_and_models = [tuple(config) for config in settings["preconfigs"]]

        optimizers = []
        for optimizer_name, defaults in settings["optimizers"].items():
            if optimizer_name == "SMB":
                for bool_ in (defaults['independent_batch']):
                    defaults_ = defaults.copy()
                    defaults_['independent_batch'] = bool_
                    optimizers.append((TestOptimizer.SMB, defaults_))
            elif optimizer_name == "SLS":
                optimizers.append((TestOptimizer.SLS, defaults))
            elif optimizer_name == "ADAM":
                optimizers.append((TestOptimizer.ADAM, defaults))
            elif optimizer_name == "SGD":
                optimizers.append((TestOptimizer.SGD, defaults))
            else:
                raise ValueError
        epochs = settings["epochs"]
        batch_size = settings.get("batch_size")
        n_samples = settings.get("n_train_samples")

        setups = []
        for dataset_and_model in datasets_and_models:
            for optimizer in optimizers:
                setups.append(
                    TrainingSetup(
                        dataset_and_model=dataset_and_model,
                        optimizer_with_config=optimizer,
                        epochs=epochs,
                        batch_size=batch_size,
                        n_train_samples=n_samples
                    )
                )

        return cls(setups=setups, seed=settings['seed'])
