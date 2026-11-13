"""Public training runner for the paper's final MoFX-Dense model."""

from __future__ import annotations

from pathlib import Path

from medunicad.configuration import (
    PAPER_SEEDS,
    PaperTrainingConfig,
    require_input_files,
)
from medunicad.training._pipeline import run_pipeline


def train(
    config: PaperTrainingConfig,
    seed: int = 42,
    resume_run_id: str | None = None,
    test_only: bool = False,
) -> None:
    """Train one paper seed or finish evaluation for an existing run."""
    config.validate_paper_settings()
    require_input_files(config)
    seed = int(seed)
    if seed not in config.seeds:
        raise ValueError(
            f"Seed {seed} is not one of the paper seeds {config.seeds}."
        )
    if resume_run_id is not None:
        resume_run_id = str(resume_run_id).strip()
        if not resume_run_id or Path(resume_run_id).name != resume_run_id:
            raise ValueError("--resume-run must be a single run-directory name.")
        run_dir = config.paths.output_dir / resume_run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        if test_only:
            best_checkpoint = run_dir / "checkpoints" / "best_model.pth"
            if not best_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Test-only checkpoint not found: {best_checkpoint}"
                )
    elif test_only:
        raise ValueError("test_only=True requires resume_run_id.")

    run_pipeline(
        data_root=str(config.paths.data_root),
        medclip_weights_path=str(config.paths.medclip_weights),
        dino_repo_path=str(config.paths.dino_repo),
        dino_cfg_path=str(config.paths.dino_config),
        dino_pretrain_path=str(config.paths.dino_checkpoint),
        use_report_label_guidance=True,
        use_cross_roi_reasoning=True,
        epochs=config.epochs,
        target_size=config.image_size,
        use_moe=True,
        resume=resume_run_id is not None,
        resume_run_id=resume_run_id,
        test_only=bool(test_only),
        ACCUM_STEPS=config.gradient_accumulation_steps,
        learning_rates=dict(config.learning_rates),
        weight_decay=config.weight_decay,
        scheduler_factor=config.scheduler_factor,
        scheduler_patience=config.scheduler_patience,
        scheduler_threshold=config.scheduler_threshold,
        mixed_precision=config.mixed_precision,
        moe_type="gradif",
        moe_num_experts=5,
        moe_top_k=2,
        gradif_attn_dim=96,
        gradif_num_heads=4,
        gradif_router_dim=128,
        gradif_top_k=2,
        gradif_routing_mode="dense_soft",
        gradif_expert_bank_type="freqxpert",
        gradif_temperature=1.5,
        gradif_temperature_end=1.5,
        gradif_temperature_hold_epochs=12,
        gradif_temperature_anneal_epochs=0,
        gradif_expert_warmup_epochs=5,
        gradif_generic_mlp_bottleneck=96,
        gradif_include_generalist=False,
        gradif_include_null_expert=True,
        gradif_modality_scale=1.0,
        gradif_router_strategy="factorized_multigate",
        gradif_router_task_residual_scale=0.10,
        gradif_router_modality_residual_scale=0.10,
        gradif_router_attention_pool=True,
        gradif_router_alpha_content_init=0.50,
        gradif_router_alpha_task_init=0.25,
        gradif_router_alpha_modality_init=0.25,
        gradif_router_task_specific_gates=True,
        gradif_router_modality_specific_gates=True,
        gradif_router_modality_content_adapters=True,
        gradif_router_branch_hidden=96,
        mofx_router_configuration="full",
        gradif_router_mi_min_group_size=4,
        gradif_router_mi_queue_size=32,
        gradif_router_mi_current_fraction=0.50,
        gradif_routing_mi_enabled=False,
        gradif_use_granularity_prior=False,
        gradif_decomposition="gaussian_fixed",
        gradif_sparsity_weight=0.0,
        gradif_load_balance_weight=0.0,
        gradif_task_separation_weight=0.0,
        gradif_expert_diversity_weight=0.0,
        gradif_gate_weight=0.0,
        gradif_router_conditioning_weight=0.0,
        gradif_logit_magnitude_weight=0.0,
        gradif_runtime_stats_every=20,
        gradif_regularization_warmup=6,
        gradif_classification_base_aux_weight=0.0,
        gradif_classification_no_harm_weight=0.0,
        gradif_segmentation_no_harm_weight=0.0,
        gradif_no_harm_start_epoch=8,
        gradif_no_harm_ramp_epochs=6,
        gradif_no_harm_every=2,
        gradif_spatial_gate_identity_weight=0.0,
        classification_gradif_fusion_init=0.25,
        classification_gradif_max_delta_ratio=0.06,
        descriptor_gt_warmup_epochs=0,
        descriptor_gt_transition_epochs=0,
        unified_batch_size=config.batch_size,
        modality_batch_counts=dict(config.modality_batch_counts),
        batches_per_epoch=config.batches_per_epoch,
        busbra_use_class_weights=True,
        birads_weight_power=0.40,
        pathology_weight_power=0.50,
        pathology_malignant_boost=1.15,
        busbra_class_weight_min=0.50,
        busbra_class_weight_max=3.00,
        llm_model_name=config.llm_model_name,
        run_name=f"mofx_dense_seed{seed}",
        experiment_seed=seed,
        deterministic_run=True,
        experiment_group_id="paper_mofx_dense",
        run_base_dir=str(config.paths.output_dir),
        run_study_name="paper_release/final_mofx_dense",
        gradif_active_stages=("c2", "c3", "c4_semantic"),
    )


def paper_seeds() -> tuple[int, ...]:
    """Return the three seeds reported in the routing comparison."""
    return PAPER_SEEDS
