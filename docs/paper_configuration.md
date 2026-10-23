# Paper configuration and code map

This release has one supported architecture and one training recipe. The table
below maps the manuscript's experiment description to the implementation.

| Manuscript setting | Release value | Implementation |
|---|---:|---|
| Input resolution | 224x224 | `configs/paper_training.json` and dataset transforms in `training/_pipeline.py` |
| Interpolation | Bicubic | dataset transforms in `training/_pipeline.py` |
| Encoder | Swin-Tiny | `modeling/freqmoe_dense.py` and `modeling/_components.py` |
| Frequency decomposition | Fixed Gaussian 3x3 and 5x5 | `FixedGaussianMoFXDecomposition` in `modeling/_components.py` |
| Frequency experts | L->H, M->H, H->M, H->L | `ArticleDirectedFreqXpert` in `modeling/_components.py` |
| Routing | Dense softmax | immutable `FINAL_ARCHITECTURE` in `modeling/freqmoe_dense.py` |
| Router context | Content, task, and modality | immutable `FINAL_ARCHITECTURE` |
| Active stages | C2, C3, C4 semantic | immutable `FINAL_ARCHITECTURE` |
| Identity route | Enabled | immutable `FINAL_ARCHITECTURE` |
| Generalist expert | Disabled | immutable `FINAL_ARCHITECTURE` |
| Epochs | 100 | `configs/paper_training.json` |
| Seeds | 42, 123, 2026 | `configs/paper_training.json` and `scripts/train.py` |
| Batch / accumulation | 8 / 4 (effective 32) | validated by `configuration.py` |
| Modality composition | 2 X-ray, 2 CT, 2 MRI, 2 ultrasound per microbatch | `EqualModalityBatchSampler` in `training/_pipeline.py` |
| Optimizer | AdamW | `create_optimizer` in `training/_pipeline.py` |
| Weight decay | 0.05 | optimizer parameter groups in `training/_pipeline.py` |
| Backbone LR | 5e-6 | optimizer schedule in `training/_pipeline.py` |
| MoFX LR | 2e-4 | optimizer schedule in `training/_pipeline.py` |
| Segmentation LR | 5e-4 | optimizer schedule in `training/_pipeline.py` |
| Classification LR | 2e-4 | optimizer schedule in `training/_pipeline.py` |
| Report LR | 1e-4 | optimizer schedule in `training/_pipeline.py` |
| Detector LR | 2e-4 | optimizer schedule in `training/_pipeline.py` |
| Report model | Llama-3.2-1B-Instruct + LoRA | configuration and `PretrainedReportDecoder` |
| Precision | Mixed precision | training loop in `training/_pipeline.py` |
| MTL optimization | Standard joint | direct summed task loss in `training/_pipeline.py` |
| Report training | Every microbatch | sampler guarantee plus a mandatory finite report-loss check in `training/_pipeline.py`; no subsampling option |
| Report metrics | ROUGE-L and METEOR | validation and test logic in `training/_pipeline.py` |
| Checkpoint selection | Mean task-family validation score, including report generation | validation/checkpoint logic in `training/_pipeline.py` |
| Geometric labels | Image/mask transforms are synchronized | joint X-ray and BUS-BRA transforms in `training/_pipeline.py` |

## Final-release execution contract

The public runner exposes only MoFX-Dense with standard joint optimization.
Alternative MoE routes, architecture experiments, expert knockouts, optimizer
ablations, routing diagnostics, and additional report metrics are not included
in the release implementation. The public constructor also rejects architecture
values that conflict with the final model.

The sampler enforces equal modality contribution at both optimization scales:
each eight-sample microbatch contains two samples per modality, and every
four-microbatch accumulation window contains eight samples per modality. MRI's
two positions rotate through full-image, segmentation, and ROI-classification
pairs; the two ultrasound positions use BI-RADS- and pathology-balanced draws.
Within that fixed 2/2/2/2 composition, the sampler guarantees at least one
valid report-supervised sample per microbatch. The training loop treats a
missing or non-finite report loss as an error instead of silently continuing.

## Training and inference agreement

Both paths instantiate `MoFXDense`. The inference configuration explicitly
matches the training values for decomposition, routing, router coefficients,
residual limits, gamma bounds, expert scaling, active stages, identity route,
and classification/segmentation fusion limits. This avoids constructing a
shape-compatible but behaviorally different model before loading a checkpoint.

Some internal keyword names retain the historical `gradif_*` prefix so that
existing finalized checkpoints load without remapping. In this repository those
keys configure MoFX-Dense only; they do not expose a second public architecture.

## Scope note

The agentic supervisor, DICOM ingestion, retrieval, and chatbot layers are not
part of this release and are planned as a later interface module. The current
API performs explicit modality/task inference with the paper model.
