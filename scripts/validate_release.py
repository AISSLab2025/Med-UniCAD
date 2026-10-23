"""Validate paper settings and Python syntax without loading model weights."""

from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from medunicad.configuration import load_training_config  # noqa: E402


def main() -> None:
    training = load_training_config(ROOT / "configs" / "paper_training.json")
    inference = json.loads(
        (ROOT / "configs" / "inference.example.json").read_text(encoding="utf-8")
    )

    expected_inference = {
        "use_moe": True,
        "moe_type": "gradif",
        "gradif_routing_mode": "dense_soft",
        "gradif_expert_bank_type": "freqxpert",
        "gradif_decomposition": "gaussian_fixed",
        "gradif_active_stages": ["c2", "c3", "c4_semantic"],
        "gradif_include_generalist": False,
        "gradif_include_null_expert": True,
        "gradif_router_strategy": "factorized_multigate",
        "gradif_use_granularity_prior": False,
        "classification_gradif_max_delta_ratio": 0.06,
        "segmentation_gradif_max_delta_ratios": [0.05, 0.04, 0.05],
    }
    mismatches = {
        key: (inference.get(key), expected)
        for key, expected in expected_inference.items()
        if inference.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Inference configuration mismatch: {mismatches}")

    runner = (SRC / "medunicad" / "training" / "runner.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "evaluation_options=",
        "enable_gradient_diagnostics=",
        "run_train_eval=",
        "resume=True",
    ):
        if forbidden in runner:
            raise ValueError(f"Final runner enables excluded experiment: {forbidden}")

    pipeline = (SRC / "medunicad" / "training" / "_pipeline.py").read_text(
        encoding="utf-8"
    )
    forbidden_pipeline_fragments = (
        "lr_report=2e-4",
        '"weight_decay": 0.0 if name == "moe_gates"',
        "rep_subsample_rate",
        "report_subsample_rate",
        "sentence_bleu",
        "bert_score_func",
        "F1RadGraph",
        "run_context_expert_knockout_ablation",
        "collect_routing_diagnostics",
        "collect_expert_validation",
        "class FreqXpertExpertValidationAccumulator",
        "class GradifRoutingAccumulator",
        "save_gradif_diagnostics",
        'self.moe_type == "evidence"',
        'self.moe_type == "sparse"',
        "sparse_top2",
        "pcgrad_merge",
        "mgda_merge",
        "cagrad_merge",
        "GradNormController",
        "gradient_surgery",
        "MRIFixedCompositionBatchSampler",
    )
    for forbidden in forbidden_pipeline_fragments:
        if forbidden in pipeline:
            raise ValueError(
                f"Training pipeline contradicts the paper release: {forbidden}"
            )

    required_runtime_fragments = (
        'lr_backbone=paper_lrs["backbone"]',
        'lr_report=paper_lrs["report"]',
        "weight_decay=weight_decay",
        '"routing_mode": "dense_soft"',
        '"expert_bank": "freqxpert"',
        '"router": "factorized_multigate"',
        'excel_row[f"{ds}_METEOR"]',
        'excel_row[f"{ds}_ROUGE_L"]',
        "class EqualModalityBatchSampler",
        "modality_batch_counts=dict(config.modality_batch_counts)",
        "batch = self._ensure_report_supervision(batch)",
        "Report loss is mandatory on every microbatch",
    )
    combined_runtime_source = pipeline + "\n" + runner
    missing_runtime_guards = [
        fragment
        for fragment in required_runtime_fragments
        if fragment not in combined_runtime_source
    ]
    if missing_runtime_guards:
        raise ValueError(
            "Release runtime is not fully wired to paper settings: "
            + ", ".join(missing_runtime_guards)
        )

    python_files = sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )
    for path in python_files:
        py_compile.compile(str(path), doraise=True)

    print(
        "Release validation passed: "
        f"{training.epochs} epochs, seeds={training.seeds}, "
        f"effective_batch_size={training.effective_batch_size}, "
        f"modality_batch_counts={dict(training.modality_batch_counts)}, "
        "report_training=every_microbatch, "
        f"compiled_files={len(python_files)}"
    )


if __name__ == "__main__":
    main()
