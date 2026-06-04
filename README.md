# Explanation-Guided Bias Mitigation in Visual Gender Classification

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

This repository contains the implementation, datasets, and experiments for our research on **Explanatory Interactive Learning (XIL)** strategies for bias mitigation in visual gender classification.

> *This repository is an extended version of our previously published CBMI paper:*
> **"Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification"**
> [IEEE Xplore – https://ieeexplore.ieee.org/document/11339339](https://ieeexplore.ieee.org/document/11339339)
>
> *This extended work builds upon the original framework with:*
> 1. **Bounding-box feedback** — comparing region-based feedback via bounding boxes versus segmentation masks
> 2. **FACET cross-dataset generalization** — out-of-distribution evaluation on the FACET benchmark
> 3. **SAM sub-region saliency analysis** — fine-grained saliency analysis over body, clothing, and hair regions using SAM-derived masks
> 4. **Multi-seed statistical robustness** — 3-seed evaluation with bootstrap confidence intervals for all configurations

We explore two XIL strategies — **CAIPI** and **Right for the Right Reasons (RRR)** — and propose a **novel hybrid approach** to guide visual classifiers toward more relevant features, reducing bias and improving fairness in gender classification tasks.

## Repository Structure

```
xil-gender-classification/
├── README.md                              # This file
├── LICENSE                                # GNU GPL v3
├── requirements.txt                       # Python dependencies
├── explanatory_gender_classification/     # Core package
│   ├── run.py                             # Quick-start training script
│   ├── run_example.py                     # Example usage walkthrough
│   ├── requirements.txt                   # Package-specific dependencies
│   ├── test_data_loading.py               # Data loading tests
│   ├── test_implementation.py             # Implementation tests
│   ├── verify_data.py                     # Dataset integrity verification
│   ├── notebooks/
│   │   └── demo.ipynb                     # Interactive demo notebook
│   ├── scripts/
│   │   ├── run_all_experiments.py         # Full experimental pipeline
│   │   ├── train_baseline.py              # Baseline model training
│   │   ├── train_caipi.py                 # CAIPI XIL training
│   │   ├── train_rrr.py                   # RRR XIL training
│   │   ├── train_hybrid.py               # Hybrid XIL training
│   │   ├── evaluate_models.py            # Model evaluation & metrics
│   │   ├── run_bbox_feedback_experiments.py  # Bbox feedback experiments
│   │   ├── run_facet_evaluation.py        # FACET OOD evaluation
│   │   ├── run_facet_mask_evaluation.py   # FACET SAM mask evaluation
│   │   ├── run_distractor_analysis.py     # Distractor region analysis
│   │   ├── prepare_facet.py               # FACET dataset preparation
│   │   ├── export_facet_labeled_images.py # Export labeled FACET images
│   │   ├── generate_explanations.py       # GradCAM/LIME explanations
│   │   ├── visualize_coco_explanations.py # COCO qualitative visualization
│   │   └── visualize_facet_explanations.py # FACET qualitative visualization
│   └── src/                               # Source modules
│       ├── augmentation/                  # Data augmentation strategies
│       ├── data/                          # Dataset loading & processing
│       ├── evaluation/                    # Metrics & bias evaluation
│       ├── explainability/                # GradCAM, BLA, LIME
│       ├── models/                        # Model architectures
│       ├── training/                      # Training loops (baseline, CAIPI, RRR, hybrid)
│       └── utils/                         # Shared utilities
├── gender_dataset/                        # COCO-based gender classification dataset
│   └── dataset_split/                     # Train/val/test split definitions
├── facet_dataset/                         # FACET benchmark dataset
│   ├── facet_single_person_binary.csv     # Filtered single-person labels
│   └── annotations/                      # FACET annotations & COCO boxes
├── results/                               # Experiment results (key CSVs only)
│   └── journal_extension/
│       ├── coco_bbox/results.csv          # Main COCO results
│       ├── facet_eval/facet_test_results.csv  # FACET evaluation results
│       └── facet_intersectional_results.csv   # Intersectional analysis
└── archived/                              # Auxiliary scripts (see archived/README.md)
```

## Requirements

- Python >= 3.8
- CUDA-capable GPU recommended (training is CPU-compatible but slow)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare datasets

Ensure the `gender_dataset/` directory contains the COCO-based gender classification images and masks. For FACET evaluation:

```bash
cd explanatory_gender_classification
python scripts/prepare_facet.py --facet_dir ../facet_dataset
```

### 2. Run all experiments

To reproduce the full experimental pipeline (all methods, explainers, seeds):

```bash
python scripts/run_all_experiments.py
```

### 3. Run individual training scripts

```bash
# Baseline (no XIL)
python scripts/train_baseline.py --explainer gradcam --seed 42

# CAIPI
python scripts/train_caipi.py --explainer gradcam --seed 42

# RRR
python scripts/train_rrr.py --explainer gradcam --seed 42

# Hybrid (CAIPI + RRR)
python scripts/train_hybrid.py --explainer gradcam --seed 42
```

### 4. Evaluate models

```bash
python scripts/evaluate_models.py
```

### 5. FACET cross-dataset evaluation

```bash
python scripts/run_facet_evaluation.py
python scripts/run_facet_mask_evaluation.py
```

## Experiments

The experimental setup evaluates **4 training methods** across **2 explainability methods** and **2 feedback types**:

| Method | Description |
|--------|-------------|
| **Baseline** | Standard training without XIL feedback |
| **CAIPI** | Counterexample-guided augmentation with corrective masks |
| **RRR** | Right for the Right Reasons — explanation regularization |
| **Hybrid** | Combined CAIPI augmentation + RRR regularization |

**Explainability methods:** GradCAM, Bounded Logit Attention (BLA)

**Feedback types:** Segmentation masks, Bounding boxes

**Evaluation:**
- **In-distribution:** COCO test set — balanced accuracy, error rate gap (ERR_gap), body saliency ratio (BSR)
- **Out-of-distribution:** FACET benchmark — cross-dataset generalization, intersectional fairness (gender × skin tone)

All configurations are evaluated with **3 random seeds** and reported with bootstrap 95% confidence intervals.

## Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{11339339,
  author    = {Satriani, Nathanya Queby and Slijep\v{c}evi\'{c}, Djordje and Schedl, Markus and Zeppelzauer, Matthias},
  booktitle = {2025 International Conference on Content-Based Multimedia Indexing (CBMI)},
  title     = {Explanatory Interactive Machine Learning for Bias Mitigation in Visual Gender Classification},
  year      = {2025},
  pages     = {1-8},
  doi       = {10.1109/CBMI66578.2025.11339339}
}
```

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
