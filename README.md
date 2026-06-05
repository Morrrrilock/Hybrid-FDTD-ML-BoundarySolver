# Hybrid-FDTD-ML-BoundarySolver
A collection of hybrid FDTD–machine methods for acoustic boundary modeling, including constrained neural boundary correction and transformer-based reflection coefficient prediction.

# Overview
This repository explores how machine learning can be integrated into finite-difference time-domain (FDTD) solvers to improve the modeling of complex acoustic boundary conditions.

Traditional FDTD simulations typically employ simplified boundary assumptions with fixed reflection coefficients. While computationally efficient, these approximations often fail to capture nonlinear and frequency-dependent boundary behavior observed in realistic acoustic systems.

To address this limitation, this repository investigates two generations of hybrid FDTD–ML methods:

- 1.A physics-constrained neural boundary predictor that directly estimates boundary pressure.
- 2.An advanced Transformer-based reflection predictor that learns dynamic reflection coefficients using long-term temporal information.

Together, these models demonstrate the evolution from local boundary correction toward memory-aware and nonlinear boundary modeling.
