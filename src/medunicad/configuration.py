"""Validated configuration objects for reproducing the paper model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PAPER_SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class TrainingPaths:
    data_root: Path
    medclip_weights: Path
    dino_repo: Path
    dino_config: Path
    dino_checkpoint: Path
    output_dir: Path


@dataclass(frozen=True)
class PaperTrainingConfig:
    paths: TrainingPaths
    epochs: int
    seeds: tuple[int, ...]
    image_size: tuple[int, int]
    batch_size: int
    modality_batch_counts: Mapping[str, int]
    gradient_accumulation_steps: int
    batches_per_epoch: int
    weight_decay: float
    learning_rates: Mapping[str, float]
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    llm_model_name: str
    mixed_precision: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def validate_paper_settings(self) -> None:
        expected_lrs = {
            "backbone": 5e-6,
            "mofx": 2e-4,
            "segmentation": 5e-4,
            "classification": 2e-4,
            "report": 1e-4,
            "detector": 2e-4,
        }
        checks = {
            "epochs": (self.epochs, 100),
            "seeds": (self.seeds, PAPER_SEEDS),
            "image_size": (self.image_size, (224, 224)),
            "batch_size": (self.batch_size, 8),
            "modality_batch_counts": (
                dict(self.modality_batch_counts),
                {"xray": 2, "ct": 2, "mri": 2, "ultrasound": 2},
            ),
            "gradient_accumulation_steps": (
                self.gradient_accumulation_steps,
                4,
            ),
            "effective_batch_size": (self.effective_batch_size, 32),
            "weight_decay": (self.weight_decay, 0.05),
            "scheduler_factor": (self.scheduler_factor, 0.5),
            "scheduler_patience": (self.scheduler_patience, 8),
            "scheduler_threshold": (self.scheduler_threshold, 1e-4),
            "llm_model_name": (
                self.llm_model_name,
                "meta-llama/Llama-3.2-1B-Instruct",
            ),
            "mixed_precision": (self.mixed_precision, True),
        }
        errors = [
            f"{name}: got {actual!r}, expected {expected!r}"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        for name, expected in expected_lrs.items():
            actual = self.learning_rates.get(name)
            if actual != expected:
                errors.append(
                    f"learning_rates.{name}: got {actual!r}, expected {expected!r}"
                )
        if errors:
            raise ValueError(
                "Configuration does not match the paper:\n- " + "\n- ".join(errors)
            )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_training_config(path: str | Path) -> PaperTrainingConfig:
    config_path = Path(path).expanduser().resolve(strict=True)
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    paths = raw["paths"]
    training = raw["training"]
    config = PaperTrainingConfig(
        paths=TrainingPaths(
            data_root=_resolve(base, paths["data_root"]),
            medclip_weights=_resolve(base, paths["medclip_weights"]),
            dino_repo=_resolve(base, paths["dino_repo"]),
            dino_config=_resolve(base, paths["dino_config"]),
            dino_checkpoint=_resolve(base, paths["dino_checkpoint"]),
            output_dir=_resolve(base, paths["output_dir"]),
        ),
        epochs=int(training["epochs"]),
        seeds=tuple(int(seed) for seed in training["seeds"]),
        image_size=tuple(int(v) for v in training["image_size"]),
        batch_size=int(training["batch_size"]),
        modality_batch_counts={
            str(name).lower(): int(value)
            for name, value in training["modality_batch_counts"].items()
        },
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        batches_per_epoch=int(training["batches_per_epoch"]),
        weight_decay=float(training["weight_decay"]),
        learning_rates={
            name: float(value)
            for name, value in training["learning_rates"].items()
        },
        scheduler_factor=float(training["scheduler"]["factor"]),
        scheduler_patience=int(training["scheduler"]["patience"]),
        scheduler_threshold=float(training["scheduler"]["threshold"]),
        llm_model_name=str(training["llm_model_name"]),
        mixed_precision=bool(training["mixed_precision"]),
    )
    config.validate_paper_settings()
    return config


def require_input_files(config: PaperTrainingConfig) -> None:
    missing = [
        str(path)
        for path in (
            config.paths.data_root,
            config.paths.medclip_weights,
            config.paths.dino_repo,
            config.paths.dino_config,
            config.paths.dino_checkpoint,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required training inputs are missing:\n- " + "\n- ".join(missing)
        )
