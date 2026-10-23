"""Final MoFX-Dense architecture used in the Med-UniCAD paper.

The wrapper fixes every architecture-defining choice while delegating to the
checkpoint-compatible implementation in ``_components``. This prevents a
published run from accidentally selecting one of the experimental variants.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from medunicad.modeling._components import SwinMultiTask as _SwinMultiTask
from medunicad.modeling._components import SwinTransformer


FINAL_ARCHITECTURE = MappingProxyType({
    "use_moe": True,
    "moe_type": "gradif",
    "moe_num_experts": 5,
    "gradif_top_k": 2,
    "gradif_routing_mode": "dense_soft",
    "gradif_expert_bank_type": "freqxpert",
    "gradif_temperature": 1.5,
    "gradif_modality_scale": 1.0,
    "gradif_router_strategy": "factorized_multigate",
    "gradif_router_task_residual_scale": 0.10,
    "gradif_router_modality_residual_scale": 0.10,
    "gradif_router_attention_pool": True,
    "gradif_router_alpha_content_init": 0.50,
    "gradif_router_alpha_task_init": 0.25,
    "gradif_router_alpha_modality_init": 0.25,
    "gradif_router_task_specific_gates": True,
    "gradif_router_modality_specific_gates": True,
    "gradif_router_modality_content_adapters": True,
    "gradif_router_branch_hidden": 96,
    "mofx_router_configuration": "full",
    "gradif_router_mi_min_group_size": 4,
    "gradif_router_mi_queue_size": 32,
    "gradif_router_mi_current_fraction": 0.50,
    "gradif_gamma_init": 0.015,
    "gradif_gamma_min": 0.0,
    "gradif_gamma_max": 0.12,
    "gradif_expert_scale_init": 0.35,
    "gradif_expert_scale_max": 1.0,
    "gradif_generic_mlp_bottleneck": 96,
    "gradif_include_generalist": False,
    "gradif_include_null_expert": True,
    "gradif_use_granularity_prior": False,
    "gradif_decomposition": "gaussian_fixed",
    "gradif_active_stages": ("c2", "c3", "c4_semantic"),
    "gradif_seg_safe_branch": True,
    "segmentation_gradif_fusion_init": 0.85,
    "segmentation_gradif_max_delta_ratios": (0.05, 0.04, 0.05),
    "classification_gradif_fusion_init": 0.25,
    "classification_gradif_max_delta_ratio": 0.06,
    "use_cross_roi_reasoning": True,
    "use_gt_descriptors": False,
})


class MoFXDense(_SwinMultiTask):
    """Paper model with immutable MoFX-Dense architecture choices.

    Dataset-dependent dimensions, checkpoint paths, and bias initializers remain
    caller supplied. Passing a conflicting architecture value raises an error
    instead of silently constructing a non-paper variant.
    """

    def __init__(self, swin_model: SwinTransformer, **kwargs: Any) -> None:
        for name, expected in FINAL_ARCHITECTURE.items():
            if name in kwargs and kwargs[name] != expected:
                raise ValueError(
                    f"{name}={kwargs[name]!r} conflicts with the final "
                    f"MoFX-Dense value {expected!r}."
                )
        merged = dict(FINAL_ARCHITECTURE)
        merged.update(kwargs)
        super().__init__(swin_model=swin_model, **merged)


# Backward-compatible aliases do not alter checkpoint state-dict keys.
FreqMoEDense = MoFXDense
SwinMultiTask = MoFXDense

__all__ = [
    "FINAL_ARCHITECTURE",
    "MoFXDense",
    "FreqMoEDense",
    "SwinMultiTask",
    "SwinTransformer",
]
