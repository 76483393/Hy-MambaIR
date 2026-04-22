# Hy-MambaIR

A hybrid Mamba-Transformer framework for cross-scale digital rock reconstruction in strongly heterogeneous conglomerates.

## Overview

Hy-MambaIR is a hybrid Mamba-Transformer framework for cross-scale digital rock reconstruction in strongly heterogeneous conglomerates. The method is designed to improve both image-level reconstruction fidelity and physically meaningful structural consistency for engineering-oriented characterization.

The framework follows a local-to-global serial design:

- a local Transformer branch captures fine pore textures and boundary details
- an attentive state-space branch models long-range structural dependencies
- a cascaded structural refinement block (SCAB) enhances structural fidelity on global feature representations

## Repository Contents

This repository includes:

- the Hy-MambaIR implementation
- training and evaluation scripts
- configuration files for training and evaluation
- one released checkpoint
- representative demo LR/HR patch pairs
- environment and dependency specifications

This repository does **not** include raw full micro-CT volumes or restricted source datasets.

## Method Components

The current implementation uses the following module terminology:

- **H-ASSB** (`HASSB` in code): hierarchical local-to-global hybrid block
- **H-SSM** (`HSSM` in code): attentive state-space global modeling branch
- **SCAB** (`SCAB` in code): spatial-channel attention refinement block
- **DyT** (`DyT` in code): optional local-branch normalization variant

## Environment

Validated local environment:

- Python 3.10.19
- PyTorch 2.10.0+cu128
- CUDA 12.8

Key dependencies:

- basicsr
- mamba_ssm
- causal_conv1d
- lpips
- timm

You can recreate the environment using either:

- `environment.yml`
- `requirements.txt`

## Repository Structure

```text
Hy-MambaIR/
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── .gitignore
├── .gitattributes
├── configs/
├── scripts/
├── core/
├── checkpoints/
└── demo_data/
```

## Quick Start

### 1. Install dependencies

Using Conda:

```bash
conda env create -f environment.yml
conda activate hymambair
```

Using pip:

```bash
pip install -r requirements.txt
```

If you cloned the repository with the released checkpoint, make sure Git LFS assets are available:

```bash
git lfs install
git lfs pull
```

### 2. Run a minimal smoke test

```bash
python scripts/smoke_test.py
```

### 3. Evaluate the released checkpoint on the demo subset

```bash
python scripts/evaluate.py --config configs/eval_demo.yaml --weights checkpoints/Hy-MambaIR_x4_main.pth --output results/eval_result.json
```

## Checkpoint

The repository currently provides one released checkpoint:

- `checkpoints/Hy-MambaIR_x4_main.pth`

This checkpoint supports evaluation and smoke testing of the released x4 setup under the provided pipeline.

## Demo Data

The repository includes four representative LR/HR demo patch pairs under:

- `demo_data/LR/`
- `demo_data/HR/`

These files are provided as a minimal reproducibility subset and do not replace the full dataset.

## Data Availability

Raw micro-CT data are not included in this repository. Access may be subject to institutional approval and applicable data-use restrictions.

## Citation

Citation information will be provided after peer review.

## Contact

For technical questions regarding code usage, checkpoint issues, or dataset access, please contact the project maintainers.
