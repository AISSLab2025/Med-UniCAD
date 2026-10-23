from __future__ import annotations

import ast
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "src" / "medunicad" / "training" / "_pipeline.py"


class _Sampler:
    pass


def _load_sampler_class():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "EqualModalityBatchSampler"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Sampler": _Sampler, "os": os, "random": random}
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["EqualModalityBatchSampler"]


def _record(index, dataset, modality, tasks, annotations=None):
    return {
        "ds_name": dataset,
        "mod_name": modality,
        "tasks": list(tasks),
        "img_path": f"image_{index}.png",
        "record": {"annotations": annotations or {}},
    }


def _fake_dataset():
    records = []
    for index in range(4):
        records.append(_record(index, "CXR", "XRay", ["classification"]))
    for index in range(4, 8):
        records.append(_record(index, "HeadCT", "CT", ["classification"]))
    for index in range(8, 11):
        records.append(
            _record(index, "LSS_MRI_AISSLab", "MRI", ["detection", "report"])
        )
    for index in range(11, 14):
        records.append(
            _record(index, "LSS_MRI_AISSLab", "MRI", ["segmentation"])
        )
    for index in range(14, 20):
        records.append(
            _record(
                index,
                "LSS_MRI_ROI_CLS",
                "MRI",
                ["classification"],
                {"classification_label": (index - 14) % 3},
            )
        )
    for index in range(20, 28):
        records.append(
            _record(
                index,
                "BUS_BRA",
                "Ultrasound",
                ["segmentation", "classification", "report"],
                {
                    "birads_classification": {"class_index": (index - 20) % 4},
                    "pathology_classification": {"class_index": (index - 20) % 2},
                },
            )
        )
    return type("Dataset", (), {"data": records})()


def _modality(record):
    if record["ds_name"] == "BUS_BRA":
        return "ultrasound"
    if record["ds_name"].startswith("LSS_MRI"):
        return "mri"
    return record["mod_name"].lower()


def test_every_microbatch_and_accumulation_window_are_modality_balanced() -> None:
    sampler_class = _load_sampler_class()
    dataset = _fake_dataset()
    sampler = sampler_class(
        dataset,
        batch_size=8,
        modality_batch_counts={
            "xray": 2,
            "ct": 2,
            "mri": 2,
            "ultrasound": 2,
        },
        batches_per_epoch=6,
        shuffle=False,
        random_seed=42,
    )
    batches = list(sampler)

    for batch in batches:
        counts = {name: 0 for name in ("xray", "ct", "mri", "ultrasound")}
        for index in batch:
            counts[_modality(dataset.data[index])] += 1
        assert counts == {"xray": 2, "ct": 2, "mri": 2, "ultrasound": 2}
        assert any(
            any("report" in task for task in dataset.data[index]["tasks"])
            for index in batch
        )

    accumulated = {name: 0 for name in ("xray", "ct", "mri", "ultrasound")}
    for batch in batches[:4]:
        for index in batch:
            accumulated[_modality(dataset.data[index])] += 1
    assert accumulated == {"xray": 8, "ct": 8, "mri": 8, "ultrasound": 8}


def test_mri_subpools_rotate_evenly_over_three_batches() -> None:
    sampler_class = _load_sampler_class()
    dataset = _fake_dataset()
    sampler = sampler_class(
        dataset,
        batch_size=8,
        batches_per_epoch=3,
        shuffle=False,
        random_seed=42,
    )
    category_counts = {"full": 0, "seg": 0, "roi": 0}

    for batch in sampler:
        for index in batch:
            record = dataset.data[index]
            if record["ds_name"] == "LSS_MRI_ROI_CLS":
                category_counts["roi"] += 1
            elif record["ds_name"] == "LSS_MRI_AISSLab":
                category_counts[
                    "seg" if "segmentation" in record["tasks"] else "full"
                ] += 1

    assert category_counts == {"full": 2, "seg": 2, "roi": 2}


def test_report_supervision_is_injected_without_changing_modality_counts() -> None:
    sampler_class = _load_sampler_class()
    dataset = _fake_dataset()
    for record in dataset.data:
        record["tasks"] = [
            task for task in record["tasks"] if "report" not in task
        ]
    dataset.data[0]["tasks"].append("report_generation")

    sampler = sampler_class(
        dataset,
        batch_size=8,
        batches_per_epoch=4,
        shuffle=False,
        random_seed=42,
    )

    for batch in sampler:
        counts = {name: 0 for name in ("xray", "ct", "mri", "ultrasound")}
        for index in batch:
            counts[_modality(dataset.data[index])] += 1
        assert counts == {"xray": 2, "ct": 2, "mri": 2, "ultrasound": 2}
        assert any(
            "report_generation" in dataset.data[index]["tasks"]
            for index in batch
        )
