import pytest
from hydra import compose, initialize_config_dir

from eqcompute.factories import MODELS, construct
from eqcompute.models.iterative import build_egnn


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_dim", 1),
        ("num_rbf", 1),
        ("graph.cutoff", 0),
        ("execution.depth", 0),
        ("execution.depth", 1.5),
        ("head.pooling", "unsupported"),
        ("weight_tying", "yes"),
    ],
)
def test_invalid_model_configs(config, key, value):
    from omegaconf import OmegaConf

    OmegaConf.update(config.model, key, value)
    with pytest.raises(ValueError):
        build_egnn(config.model, "mu")


def test_presets_and_factory():
    from pathlib import Path

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        config = compose(config_name="config", overrides=["experiment=qm9_loop_egnn_depth"])
        assert config.model.weight_tying is True
        assert config.model.execution.depth == 4
        assert config.task.supervision == "final"
        assert "tiedTrue_k4_final" in config.experiment_name
    with pytest.raises(ValueError, match="Unknown component"):
        construct(MODELS, "adaloop")
