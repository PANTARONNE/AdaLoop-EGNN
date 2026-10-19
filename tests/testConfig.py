import pytest
from hydra import compose, initialize_config_dir

from eqcompute.factories import DATASETS, MODELS, construct
from eqcompute.models.iterative import build_egnn


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_dim", 1),
        ("execution.depth", 0),
        ("execution.depth", 1.5),
        ("execution.name", "unsupported"),
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
        config = compose(config_name="config", overrides=["experiment=qm9_loop_egnn"])
        assert config.model.weight_tying is True
        assert config.model.execution.depth == 7
        assert config.task.supervision == "final"
        assert "tiedTrue_k7_final" in config.experiment_name
    with pytest.raises(ValueError, match="Unknown component"):
        construct(MODELS, "adaloop")


@pytest.mark.parametrize("preset", [
    "qm9_smoke", "qm9_egnn", "qm9_loop_egnn", "qm9_loop_egnn_diagnostic",
    "qm9_adaloop_egnn", "qm9_adaloop_egnn_smoke",
])
def test_egnn_backbone_presets_use_local_pyg_qm9(preset):
    from pathlib import Path

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        config = compose(config_name="config", overrides=[f"experiment={preset}"])
    assert config.dataset.name == "qm9"
    assert config.dataset.root == "data/qm9"
    assert config.model.name == "egnn"
    assert not {"num_rbf", "coordinate_scale", "graph", "head"} & set(config.model)
    construct(MODELS, config.model.name, config.model, config.task.target)
    assert config.task.normalization == "mad"
    assert "subtract_thermo" not in config.dataset
    if preset.endswith("smoke"):
        assert (config.dataset.train_size, config.dataset.val_size, config.dataset.test_size) == (
            128, 32, 32
        )
        assert config.dataset.subset_size == 192
        assert "train_limit" not in config.dataset
    else:
        assert config.dataset.train_size == 110000
        assert config.dataset.val_size == 10000
        assert config.dataset.split_seed == 42


def test_default_is_egnn_baseline():
    from pathlib import Path

    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        default = compose(config_name="config")
        baseline = compose(config_name="config", overrides=["experiment=qm9_egnn"])
    assert OmegaConf.to_container(default, resolve=True) == OmegaConf.to_container(
        baseline, resolve=True
    )
    assert default.model.name == "egnn"
    assert default.model.execution.depth == 7
    assert default.task.normalization == "mad"
    assert default.optimizer.name == "adam"
    assert default.trainer.scheduler.name == "cosine"
    default.task.target = "homo"
    assert default.optimizer.learning_rate == 0.001
    assert set(MODELS) == {"egnn"}
    assert set(DATASETS) == {"qm9"}
