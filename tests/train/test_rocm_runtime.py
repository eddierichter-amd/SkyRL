from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import get_ray_init_num_gpus


def test_ray_init_gpu_count_is_per_node(monkeypatch):
    monkeypatch.delenv("NUM_GPUS", raising=False)
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.policy_num_nodes = 3
    cfg.trainer.placement.ref_num_gpus_per_node = 2
    cfg.trainer.placement.ref_num_nodes = 3

    assert get_ray_init_num_gpus(cfg) == 4


def test_ray_init_gpu_count_allows_container_override(monkeypatch):
    monkeypatch.setenv("NUM_GPUS", "2")

    assert get_ray_init_num_gpus(SkyRLTrainConfig()) == 2
