# DPLTGAN
This repository contains a snapshot of the DPLTGAN code: a Differential Privacy Trajectory Synthesizer capable of learning both long-term and short-term temporal dependencies in trajectories. 

Overview
The Differential Privacy Trajectory Synthesizer (DP-LTGAN) is a method designed to synthesize trajectory data with differential privacy (DP) guarantees. It can learn and generate trajectories while maintaining both long-term and short-term temporal dependencies. The synthesized trajectories are generated with high utility, preserving the statistical properties of the original data, while providing privacy protection for individual data points.

Requirements
Before running the script, make sure to install the necessary dependencies. All dependencies are listed in the requirements.txt file.

To train the DPLTGAN model, simply run the following script:
python DPLTGAN_train.py

Evaluating Synthetic Data
Refer to the code at https://github.com/iWitLab/evaluate_synthetic_time_series for evaluating metrics of synthetic time series data. The script compares synthetic data with the original data and analyzes how well the synthetic data performs in protecting the privacy of the original data subjects, maintaining statistical similarity, achieving per-instance similarity, and ensuring diversity of the synthetic data.

Paper Reference
This method is described in the paper:
"Differentially Private Trajectory Publishing via Locally-aware Transformer-based GAN (DP-LTGAN)"
The paper presents the core methodology and provides theoretical foundations for the techniques implemented in this repository.
