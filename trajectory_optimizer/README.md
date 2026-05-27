# Multiphysics Environment

A computational physics repository dedicated to high-precision trajectory tracking, kinematic solvers, and machine learning-driven physical optimization.

## Project Progression

This repository represents an evolution from basic forward kinematics to advanced, high-precision inverse physics solvers utilizing deep learning frameworks (PyTorch).

* **`trajectory.py`**: Baseline forward kinematics simulating standard projectile motion using NumPy and Matplotlib.
* **`optimize.py`**: Introductory 1D PyTorch Adam optimization to find the required launch angle for a specific distance.
* **`lhtarget.py` & `lbfgs_optimizer.py`**: 2D coordinate targeting in a vacuum, demonstrating the transition from first-order (Adam) to second-order (L-BFGS) gradient descent methods.
* **`drag_optimizer.py` & `x.py`**: Advanced multiphysics simulations introducing continuous aerodynamic drag ($-kv\mathbf{v}$) and adaptive time-stepping (Courant/CFL conditions).
* **`1D_drag_optimizer.py`**: The culminating Hybrid Optimization Cascade. It locks the launch angle tactically and solves for exact initial velocity using 64-bit precision, continuous line-segment projection, and a two-phase descent (Adam for ballistic approach $\rightarrow$ L-BFGS for $10^{-12}$ precision targeting).

## Tech Stack
* Python 3.12+
* PyTorch (Autograd, Adam, L-BFGS)
* NumPy & Matplotlib
