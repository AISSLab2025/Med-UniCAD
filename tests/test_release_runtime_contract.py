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
PUBLIC_SCRIPTS = {
    name: (ROOT / "scripts" / name).read_text(encoding="utf-8")
    for name in ("train.py", "infer.py", "serve_api.py")
}


def test_public_scripts_support_direct_source_checkout_execution() -> None:
    for source in PUBLIC_SCRIPTS.values():
        assert 'SRC = ROOT / "src"' in source
        assert "sys.path.insert(0, str(SRC))" in source


def test_training_cli_supports_safe_test_only_resume() -> None:
    train_script = PUBLIC_SCRIPTS["train.py"]
    assert '"--resume-run"' in train_script
    assert '"--test-only"' in train_script
    assert "resume_run_id=args.resume_run" in train_script
    assert "test_only=args.test_only" in train_script
    assert "resume_run_id=resume_run_id" in RUNNER
    assert "test_only=bool(test_only)" in RUNNER
    assert "if test_only:" in PIPELINE
    assert 'start_epoch = int(epochs)' in PIPELINE
    assert 'loads best_model.pth directly' in PIPELINE


def test_report_guidance_is_explicitly_wired_into_datasets() -> None:
    assert "USE_REPORT_LABEL_GUIDANCE" not in PIPELINE
    assert "use_report_label_guidance = bool(use_report_label_guidance)" in PIPELINE
    assert (
        "use_report_label_guidance=use_report_label_guidance" in PIPELINE
    )


def test_cross_task_reasoning_is_explicitly_wired_into_model() -> None:
    assert "USE_CROSS_ROI_REASONING" not in PIPELINE
    assert "use_cross_roi_reasoning = bool(use_cross_roi_reasoning)" in PIPELINE
    assert "use_cross_roi_reasoning=use_cross_roi_reasoning" in PIPELINE
    assert "use_cross_roi_reasoning=True" in RUNNER


def test_dino_import_does_not_depend_on_working_directory() -> None:
    assert 'dino_repo = Path(repo_path).expanduser().resolve()' in COMPONENTS
    assert 'for import_path in (dino_repo, dino_repo.parent):' in COMPONENTS
    assert 'from DINO.util.slconfig import SLConfig' in COMPONENTS
    assert 'must be named \'DINO\'' in COMPONENTS


def test_llm_loader_uses_current_transformers_dtype_argument() -> None:
    assert "torch_dtype=" not in COMPONENTS
    assert "dtype=torch.float16" in COMPONENTS


def test_greedy_report_generation_does_not_pass_sampling_flags() -> None:
    assert "if do_sample:" in COMPONENTS
    assert "generation_kwargs.update(temperature=0.4, top_p=0.95)" in COMPONENTS


def test_mofx_runtime_diagnostic_stores_exist_before_first_forward() -> None:
    stores = (
        "latest_expert_band_rms",
        "latest_expert_band_fractions",
        "latest_expert_target_fraction",
        "latest_expert_target_dominant",
        "latest_expert_source_cosine",
        "latest_expert_target_cosine",
        "latest_expert_applied_scale",
        "latest_expert_visuals",
    )
    for store in stores:
        assert f"self.{store} = {{}}" in COMPONENTS
        assert f"self.{store}," in COMPONENTS


def test_mofx_diagnostic_intervals_are_explicit_runtime_inputs() -> None:
    assert "gradif_runtime_stats_every=20" in PIPELINE
    assert "gradif_no_harm_every=2" in PIPELINE
    assert "gradif_runtime_stats_every=20" in RUNNER
    assert "gradif_no_harm_every=2" in RUNNER


def test_current_pytorch_runtime_and_determinism_contract() -> None:
    assert 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")' in PIPELINE
    assert "torch.cuda.amp.GradScaler" not in PIPELINE
    assert "torch.cuda.amp.autocast" not in PIPELINE
    assert 'torch.amp.GradScaler("cuda", enabled=use_amp)' in PIPELINE
    assert 'torch.meshgrid(coords_h, coords_w, indexing="ij")' in COMPONENTS


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
    assert "b_score" not in PIPELINE
    assert '"bleu"' not in PIPELINE.lower()
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
