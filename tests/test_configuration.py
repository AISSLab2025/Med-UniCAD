from __future__ import annotations

import json
from pathlib import Path

from medunicad.configuration import PAPER_SEEDS, load_training_config


ROOT = Path(__file__).resolve().parents[1]


def test_paper_training_configuration() -> None:
    config_path = ROOT / "configs" / "paper_training.json"
    config = load_training_config(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert config.epochs == 100
    assert config.seeds == PAPER_SEEDS
    assert config.image_size == (224, 224)
    assert config.batch_size == 8
    assert dict(config.modality_batch_counts) == {
        "xray": 2,
        "ct": 2,
        "mri": 2,
        "ultrasound": 2,
    }
    assert config.gradient_accumulation_steps == 4
    assert config.effective_batch_size == 32
    assert config.weight_decay == 0.05
    assert "report_subsample_rate" not in raw["training"]
    assert config.learning_rates == {
        "backbone": 5e-6,
        "mofx": 2e-4,
        "segmentation": 5e-4,
        "classification": 2e-4,
        "report": 1e-4,
        "detector": 2e-4,
    }
    assert config.scheduler_factor == 0.5
    assert config.scheduler_patience == 8
    assert config.scheduler_threshold == 1e-4


def test_inference_uses_final_dense_frequency_model() -> None:
    config = json.loads(
        (ROOT / "configs" / "inference.example.json").read_text(encoding="utf-8")
    )
    assert config["gradif_routing_mode"] == "dense_soft"
    assert config["gradif_expert_bank_type"] == "freqxpert"
    assert config["gradif_decomposition"] == "gaussian_fixed"
    assert config["gradif_include_generalist"] is False
    assert config["gradif_include_null_expert"] is True
    assert config["gradif_use_granularity_prior"] is False
    assert config["classification_gradif_max_delta_ratio"] == 0.06
