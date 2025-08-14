# Multi-Task Learning for Implicit Hate Detection

This repository contains the code, datasets, and results for our EE-559 Deep Learning mini-project on **implicit hate speech detection**. We explore a **Multi-Task Learning (MTL)** framework with BERT, jointly training on sarcasm detection, stereotype detection, and fine-grained implicit hate subtype classification to improve both in-domain performance and cross-domain generalization.

## Repository Structure
- datasets/ – Datasets used in the experiments
- models/ – Saved models and checkpoints
- results/ – Evaluation scripts and outputs
- scripts/ – Training and ablation study scripts
- Other files include the project report, poster, ablation study, and experimental results.

## Main Scripts

### Training
1. scripts/mtl-main-local.py – MTL training with different auxiliary task weights (produces Table 1 in the report)
2. scripts/baseline-main-local.py – Baseline single-task training (Table 1)
3. scripts/ablation-study-local.py – Ablation study for auxiliary tasks (Table 2)

Note: Local versions are for demonstration and have not been run to completion due to memory limits.  
For full runs, use the corresponding *-cluster.py scripts.

### Testing
1. results/test-toxigen.py – Out-of-domain evaluation on ToxiGen (Table 3)
2. results/misclassification_fine_grain.py – Fine-grained misclassification analysis (Table 4)

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Update dataset paths in the scripts as needed.

3. Run the desired training or evaluation script:
   python scripts/mtl-main-cluster.py

## Project Details
- Report: Report-EE559.pdf
- Poster: Poster-EE559.pdf
- Course: EE-559 Deep Learning, EPFL
- Authors: Semanur Avşar, Ines Altemir Marinas, Florian Tanguy
