from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    ROOT / "src" / "medunicad" / "modeling" / "_components.py"
).read_text(encoding="utf-8")
PIPELINE = (
    ROOT / "src" / "medunicad" / "training" / "_pipeline.py"
).read_text(encoding="utf-8")
RUNNER = (
    ROOT / "src" / "medunicad" / "training" / "runner.py"
).read_text(encoding="utf-8")


def test_paper_optimizer_values_are_runtime_inputs() -> None:
    assert 'lr_backbone=paper_lrs["backbone"]' in PIPELINE
    assert 'lr_report=paper_lrs["report"]' in PIPELINE
    assert 'lr_moe_gates=paper_lrs["mofx"]' in PIPELINE
    assert '"weight_decay": float(weight_decay)' in PIPELINE
    assert "lr_report=2e-4" not in PIPELINE
    assert '0.0 if name == "moe_gates"' not in PIPELINE


def test_final_runner_excludes_non_paper_evaluation() -> None:
    assert "evaluation_options=" not in RUNNER
    assert "collect_routing_diagnostics" not in RUNNER
    assert "knockout_ablation" not in RUNNER
    assert "enable_gradient_diagnostics=" not in RUNNER
    assert "run_train_eval=" not in RUNNER
    assert "rep_subsample_rate" not in RUNNER
    assert "rep_subsample_rate" not in PIPELINE
    assert "Report loss is mandatory on every microbatch" in PIPELINE
    assert "Report-training contract violated" in PIPELINE
    assert "gradif_logit_magnitude_weight=0.0" in RUNNER
    assert "gradif_classification_base_aux_weight=0.0" in RUNNER
    assert "gradif_classification_no_harm_weight=0.0" in RUNNER
    assert "gradif_segmentation_no_harm_weight=0.0" in RUNNER
    assert "pcgrad_merge" not in PIPELINE
    assert "mgda_merge" not in PIPELINE
    assert "cagrad_merge" not in PIPELINE
    assert "GradNormController" not in PIPELINE
    assert "gradient_surgery" not in PIPELINE


def test_xray_segmentation_uses_joint_geometry() -> None:
    assert "class XRayJointSegTransform" in PIPELINE
    assert "image = TF.affine(" in PIPELINE
    assert "mask = TF.affine(" in PIPELINE
    assert "image = TF.resized_crop(" in PIPELINE
    assert "mask = TF.resized_crop(" in PIPELINE
    assert "is_xray_seg_joint" in PIPELINE


def test_equal_modality_sampler_is_the_only_training_sampler() -> None:
    assert "class EqualModalityBatchSampler" in PIPELINE
    assert "batch_sampler = EqualModalityBatchSampler(" in PIPELINE
    assert "MRIFixedCompositionBatchSampler" not in PIPELINE
    assert "modality_batch_counts=dict(config.modality_batch_counts)" in RUNNER
    assert "batch = self._ensure_report_supervision(batch)" in PIPELINE


def test_report_metrics_match_the_manuscript() -> None:
    assert "sentence_bleu" not in PIPELINE
    assert "bert_score_func" not in PIPELINE
    assert "F1RadGraph" not in PIPELINE
    assert 'excel_row[f"{ds}_METEOR"]' in PIPELINE
    assert 'excel_row[f"{ds}_ROUGE_L"]' in PIPELINE


def test_base_model_rejects_non_final_moe_variants() -> None:
    assert 'raise ValueError("The public model supports MoFX-Dense only.")' in COMPONENTS
    assert "class MoFXDenseBlock" in COMPONENTS
    assert "class MoFXDenseHierarchy" in COMPONENTS
    assert 'self.moe_type == "evidence"' not in COMPONENTS
    assert 'self.moe_type == "sparse"' not in COMPONENTS
    assert "sparse_top2" not in COMPONENTS
