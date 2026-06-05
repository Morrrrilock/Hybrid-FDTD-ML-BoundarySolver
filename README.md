# Hybrid-FDTD-ML-BoundarySolver
A collection of hybrid FDTD–machine methods for acoustic boundary modeling, including constrained neural boundary correction and transformer-based reflection coefficient prediction.

# Overview
This repository explores how machine learning can be integrated into finite-difference time-domain (FDTD) solvers to improve the modeling of complex acoustic boundary conditions.

Traditional FDTD simulations typically employ simplified boundary assumptions with fixed reflection coefficients. While computationally efficient, these approximations often fail to capture nonlinear and frequency-dependent boundary behavior observed in realistic acoustic systems.

To address this limitation, this repository investigates two generations of hybrid FDTD–ML methods:

- 1. A physics-constrained neural boundary predictor that directly estimates boundary pressure.
- 2. An advanced Transformer-based reflection predictor that learns dynamic reflection coefficients using long-term temporal information.

Together, these models demonstrate the evolution from local boundary correction toward memory-aware and nonlinear boundary modeling.

# Method 1: Physics-Constrained Boundary Pressure Prediction

The first-generation hybrid model focuses on learning the boundary pressure directly.

A compact neural network receives local boundary information, including:

- Neighbor pressure
- Previous pressure state
- Pressure gradient
- Simulation time
- Source frequency

and predicts the boundary pressure used by the FDTD solver.

To maintain numerical stability, several physical constraints are imposed:

- Reflection-coefficient limits
- Amplitude clipping
- Sign consistency
- Stability-preserving corrections

This model successfully learns realistic boundary behavior and significantly improves local boundary prediction accuracy compared with simplified reflection models.

However, the influence on the entire spatial pressure field remains limited because the model relies primarily on instantaneous local information and does not explicitly capture the accumulated effects of previous wave–boundary interactions.
