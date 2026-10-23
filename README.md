# Med-UniCAD

This repository contains the final MoFX-Dense implementation used for the
paper *A Unified Multimodal and Multitask Medical CAD Framework with Mixture of
FreqXperts*. The public entry points construct and execute only the final
MoFX-Dense recipe; routing baselines, expert knockouts, and alternative
multitask optimizers are never selected by the release runner.

The public model constructor fixes the paper architecture: a Swin-Tiny encoder,
same-resolution 3x3/5x5 Gaussian frequency decomposition, four directed
FreqXperts (low-to-high, mid-to-high, high-to-mid, and high-to-low), an identity
route, factorized content/task/modality conditioning, and dense softmax routing
at C2, C3, and C4.

## Repository layout

```text
configs/                 Paper training and inference configuration
docs/                    Code-to-paper map and API guide
scripts/                 Training, inference, serving, and validation CLIs
src/medunicad/
  modeling/              Final public model and checkpoint-compatible components
  training/              Training runner and complete optimization pipeline
  inference/             File/CLI inference engine and FastAPI service
  losses/                DINO matching, criterion, and box operations
tests/                   Lightweight configuration and release checks
```

## Installation

Python 3.10 or newer is required. Install in editable mode:

```bash
python -m pip install -e ".[train,api]"
```

PyTorch wheels are platform- and CUDA-specific. If the pinned default wheel is
not appropriate for your system, install the matching PyTorch build first, then
install this project.

The official DINO repository is expected under `third_party/DINO` by the
example configuration. Dataset files and pretrained checkpoints are not stored
in Git. Place them locally and update the paths in `configs/paper_training.json`
and `configs/inference.example.json`.

Llama-3.2-1B-Instruct may require accepting its provider's access terms before
the tokenizer and weights can be downloaded.

## Reproduce training

The paper comparison uses seeds 42, 123, and 2026. Run one seed per process:

```bash
python scripts/validate_release.py
python scripts/train.py --config configs/paper_training.json --seed 42
python scripts/train.py --config configs/paper_training.json --seed 123
python scripts/train.py --config configs/paper_training.json --seed 2026
```

Each run performs 100 epochs with per-step batch size 8 and four-step gradient
accumulation (effective batch size 32). Every microbatch contains two X-ray,
two CT, two MRI, and two ultrasound samples; consequently, each accumulated
batch contains eight samples from every modality. Fresh training is the default; no
experimental initializer is exposed by the public runner.

The encoder and LoRA adapters are trainable from epoch 0. Every microbatch is
guaranteed to contain at least one labelled report sample, so the complete
report path and report loss run at every training iteration. Report subsampling
is not present in the release. The fixed-composition sampler emits 500 batches
per epoch.
All optimizer learning rates, weight decay, and scheduler values are read from
`configs/paper_training.json`.

## Run inference

Copy `configs/inference.example.json` if desired, set all paths, then run:

```bash
python scripts/infer.py \
  --config configs/inference.example.json \
  --image path/to/image.png \
  --modality ultrasound \
  --tasks segmentation,birads,pathology,report
```

Supported modalities are `xray`, `ct`, `mri`, and `ultrasound`. The engine
automatically adds prerequisite tasks when a requested output depends on
segmentation or detection evidence.

For HTTP serving, see `docs/api_usage.md`.

## Reproducibility notes

- Image size: 224x224, bicubic interpolation, pretrained encoder normalization.
- Optimizer: AdamW with weight decay 0.05.
- Batch composition: 2 X-ray + 2 CT + 2 MRI + 2 ultrasound samples per
  microbatch; MRI supervision pools rotate evenly across batches.
- Learning rates: backbone 5e-6; MoFX 2e-4; segmentation 5e-4;
  classification 2e-4; report modules 1e-4; detector 2e-4.
- Report decoder: frozen Llama-3.2-1B-Instruct backbone with LoRA adaptation.
- Report training: at least one supervised report in every microbatch; no
  report-path subsampling.
- Report evaluation: ROUGE-L and METEOR.
- Checkpoint selection: mean validation score across segmentation,
  classification, detection, ROI classification, and report generation.
- Standard joint multitask optimization is used; gradient-surgery implementations
  are not included in the release pipeline.
- Random geometric augmentation is synchronized between images and their masks.

## Current scope

This repository currently covers the paper model, training pipeline, and
explicit-task inference API. The natural-language agentic supervisor, DICOM
ingestion, retrieval layer, and chatbot interface will be released separately;
they are not implemented or claimed as reproducible by this version.

See `docs/paper_configuration.md` for the exact code locations corresponding to
the manuscript description.
