# ℓ₁-Regularized Least Squares via Proximal Gradient Methods

## Overview
This project studies **first-order optimization methods** for solving the ℓ₁-regularized least squares problem (LASSO), with a focus on **proximal gradient descent (ISTA)**. The goal is to connect **convex optimization theory** with **practical algorithmic behavior**, particularly sparsity emergence and convergence dynamics.

The emphasis is on **explicit algorithm implementation and controlled numerical experiments**, rather than black-box solver usage.

---

## Problem Formulation
We consider the ℓ₁-regularized least squares problem

$$
\min_x \; \frac{1}{2}\|Ax - b\|_2^2 + \lambda \|x\|_1
$$

where:
- $A \in \mathbb{R}^{m \times n}$ is the data matrix  
- $b \in \mathbb{R}^m$ is the observation vector  
- $\lambda > 0$ controls the sparsity induced by ℓ₁ regularization  

This objective consists of a **smooth convex term** and a **non-smooth convex term**, making it well-suited for proximal optimization methods.

---

## Methodology

### Proximal Gradient Descent (ISTA)
ISTA alternates between:
1. A **gradient descent step** on the smooth least-squares term  
2. A **proximal (soft-thresholding) step** to handle the non-smooth ℓ₁ regularization  

The update rule is given by

$$
x^{k+1} = \text{soft}_{\lambda t}\left(x^k - t \nabla f(x^k)\right)
$$


where $t$ is a step size chosen based on the Lipschitz constant of the gradient of the smooth term.

---

## Experimental Setup
All experiments are conducted on **synthetic datasets** to isolate algorithmic effects.

The experimental pipeline includes:
- Randomly generated sensing matrices and sparse ground-truth vectors  
- Controlled noise injection  
- Fixed random seeds for reproducibility  

---

## Experiments and Analysis
The experiments investigate:
- **Objective value decay** across iterations  
- **Convergence behavior** under different step sizes  
- **Emergence of sparsity** as a function of the regularization parameter $\lambda$  
- Trade-offs between convergence speed and sparsity strength  

Results are visualized using plots of objective value and sparsity patterns over iterations.

---

## Key Observations
- Larger values of $\lambda$ promote stronger sparsity but slow convergence  
- ISTA exhibits stable convergence consistent with convex optimization theory  
- Soft-thresholding enforces sparsity implicitly through the proximal operator  
- Empirical behavior closely aligns with theoretical expectations  

---

## Implementation Details
- **Language:** Python  
- **Libraries:** NumPy, Matplotlib  
- Optimization routines are implemented explicitly without solver abstractions  
- All experiments are reproducible  

---

## Current Status
🚧 **Under active development**

Planned extensions:
- Accelerated proximal methods (FISTA)  
- Comparisons with coordinate descent  
- Experiments with structured sensing matrices  
- Additional convergence diagnostics  

---

## Motivation
This project aims to build intuition for **structured convex optimization** by bridging theory, algorithm design, and empirical validation, skills central to research in optimization, machine learning, and signal processing.
