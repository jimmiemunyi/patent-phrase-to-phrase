import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from fastcore.xtras import Path  # for ls

logger = logging.getLogger("TRAIN")
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # printing the configurations
    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # loading the dataset
    # logger.info("Loading the dataset")
    path = Path(cfg.data.path)
    print(path.ls())


if __name__ == "__main__":
    train()
