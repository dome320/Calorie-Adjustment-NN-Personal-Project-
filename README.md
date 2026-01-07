# Adaptive Calorie Adjustment Neural Network (PyTorch)

This project is a personal, experimental neural network built in **PyTorch** that predicts **weekly calorie adjustments** based on recent physiological feedback.

The model is designed to mimic the decision-making process of a human coach by analyzing:
- Recent calorie intake
- Short-term bodyweight trends
- A target rate of weight change

It outputs a **discrete calorie adjustment** (e.g. `-300`, `-100`, `0`, `+200`, `+300` kcal/day) rather than a continuous value, emphasizing **actionable decisions** over raw prediction.

---

## Motivation

Most calorie calculators are static and fail to adapt to real-world feedback.

This project explores:
- How neural networks can act as **adaptive controllers**
- Translating physiological signals into algorithmic decisions
- Building an end-to-end ML pipeline from data → training → inference

The long-term goal is to iteratively improve this system and document experimentation with:
- Feature engineering
- Model architecture
- Training strategies
- Evaluation methods

This repository serves both as a **learning log** and a foundation for more advanced experimentation.

---

## Problem Framing

**Inputs (weekly):**
1. Average daily calories from the previous week
2. Weekly bodyweight change (lbs)
3. Target rate of weight change (lbs/week)

**Output:**
- One of **7 discrete calorie adjustments**  
  `{ -300, -200, -100, 0, +100, +200, +300 } kcal/day`

This framing treats the problem as a **multi-class classification task**, prioritizing stability and interpretability over continuous regression.

---

## Model Overview

- Framework: **PyTorch**
- Architecture: Fully connected MLP
  - Input layer: 3 features
  - Hidden layers: 2
  - Output layer: 7 classes
- Activation: ReLU
- Loss function: CrossEntropyLoss
- Optimizer: Adam

The model is intentionally simple to make experimentation and iteration fast and interpretable.

---

## Project Structure

NN Project/
├── nn.py # Model definition, training, and inference
├── bulk_cut_dataset.csv # Synthetic training dataset
├── calorie_nn.pt # Saved trained model weights
├── .gitignore
└── README.md


## Example Interaction

Enter weekly inputs:
cal_avg_last_week: 2900
bw_change_last_week: -0.6
target_rate_lbs_per_week: -0.5

Predicted calorie adjustment: -100 kcal/day

