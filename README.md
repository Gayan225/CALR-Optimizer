
# Curvature-Adaptive Learning Rate Optimizer (CALR)
This repository contains the code and supplementary materials for our FLAIRS-38 paper:

**Curvature-Adaptive Learning Rate Optimizer: Theoretical Insights and Empirical Evaluation on Neural Network Training**  
📌 *Kehelwala Dewage Gayan Maduranga*  

## 📌 Overview
CALR is an adaptive optimization method that dynamically adjusts the learning rate based on local curvature estimates. This repository includes:
- 🔹 Implementation of CALR
- 🔹 Experiments on synthetic benchmark functions (Rosenbrock, Himmelblau, Saddle Point)
- 🔹 Neural network training on MNIST and CIFAR-10
- 🔹 Supplementary materials and extended analyses beyond the 6-page paper limit

## 🚀 Getting Started
### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/CALR-Optimizer.git
cd CALR-Optimizer
pip install -r requirements.txt
📖 Citation
If you use this code, please cite:
@inproceedings{maduranga2025calr,
  author    = {Kehelwala Dewage Gayan Maduranga},
  title     = {Curvature-Adaptive Learning Rate Optimizer: Theoretical Insights and Empirical Evaluation on Neural Network Training},
  booktitle = {FLAIRS-38},
  year      = {2025}
}
