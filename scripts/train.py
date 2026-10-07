import hydra
from omegaconf import DictConfig

from eqcompute.application import train


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()
