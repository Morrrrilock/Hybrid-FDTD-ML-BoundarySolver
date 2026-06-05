# Hybrid-FDTD-ML-BoundarySolver
A collection of hybrid FDTD–machine methods for acoustic boundary modeling, including constrained neural boundary correction and transformer-based reflection coefficient prediction.

# Overview
This repository explores how machine learning can be integrated into finite-difference time-domain (FDTD) solvers to improve the modeling of complex acoustic boundary conditions.

Traditional FDTD simulations typically employ simplified boundary assumptions with fixed reflection coefficients. While computationally efficient, these approximations often fail to capture nonlinear and frequency-dependent boundary behavior observed in realistic acoustic systems.

To address this limitation, this repository investigates two generations of hybrid FDTD–ML methods:

- A physics-constrained neural boundary predictor that directly estimates boundary pressure.
- An advanced Transformer-based reflection predictor that learns dynamic reflection coefficients using long-term temporal information.

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
# Method 2: Transformer-Based Reflection Coefficient Prediction

The second-generation model extends the previous framework by introducing long-term memory and nonlinear boundary modeling.

Instead of directly predicting boundary pressure, the model predicts the reflection coefficient governing the boundary update.

The architecture combines:

- Multi-head self-attention
- Deep residual networks
- Historical state encoding
- Learnable physical priors
- Reflection-coefficient constraints

The model incorporates:

- Hidden dimension: 512
- 6 Transformer layers
- 8 residual blocks
- 8-head self-attention
- 30-step history window
- Approximately 2.5 million trainable parameters
- 
Unlike the first-generation model, which primarily relies on instantaneous local information, the Transformer-based approach explicitly incorporates historical boundary states through a memory window. This enables the network to capture delayed reflections and accumulated wave–boundary interactions that cannot be represented by local features alone.

The self-attention mechanism further allows the model to identify relationships between distant temporal states and extract global temporal patterns from the boundary history. In addition, the deep nonlinear architecture provides a more flexible representation of frequency-dependent and amplitude-dependent boundary behavior.

By learning reflection coefficients rather than boundary pressure directly, the model offers a more physically interpretable description of the boundary dynamics while preserving compatibility with the underlying FDTD solver.

As a result, the Transformer model produces more realistic boundary responses and achieves higher boundary prediction accuracy than the first-generation approach. The improved boundary representation also leads to a reduction in full-field spatial error, although the improvement in the overall wavefield remains more limited than the improvement observed at the boundary itself.
