# Optimal control for sampling the transition path process and estimating rates

This repository contains the implementation of paper **Optimal control for sampling the transition path process and estimating rates**. 

We consider Ito's diffusion process with rare transitions between two predefined sets $A$ and $B$:

$$ dX_t = b(X_t) dt + \sigma(X_t) dW_t. $$

In particular, we focus on two instances of it: Langevin dynamics and overdamped Langevin dynamics in Collective Variables(CVs) governed by 
SDEs shown as follows

Langevin dynamics: 

$$ \begin{cases}
        dX_t = m^{-1} P_t dt \\
        dP_t = - (\nabla U + \gamma_f P_t) dt + \sqrt{2 \gamma_f \beta^{-1} m} dW_t,
    \end{cases} $$
    
Overdamped Langevin dynamics in CVs: 

$$ d\xi_t = [-M(\xi_t)\nabla F(\xi_t) + \beta^{-1}\nabla \cdot M(\xi_t)]dt + \sqrt{2\beta^{-1}}M(\xi_t)^{\frac{1}{2}}dW_t. $$

Under the optimally controlled process

$$ dY_t = (b(Y_t) + \sigma(Y_t) \sigma^T(Y_t) v^*(Y_t)) dt + \sigma(Y_t) dW_t $$

such that $v^*(\cdot)$ satisfies

$$ \sigma^\top v^* = \sigma^\top \nabla \log q^+,$$

trajectories starting from boundary of $A$ goes to $B$ almost sure, where $q^+$ is the forward committor function. 

The transition rate from $A$ to $B$ is defined as

$$ \nu_{AB} = \lim_{T \rightarrow \infty} \frac{N_{AB}}{T}$$

where $N_{AB}$ denotes the number of transitions from $A$ to $B$ in time interval $[0,T]$. The transition rate can be recovered via

$$ \nu_{AB} = \frac{\rho_{AB}}{\mathbb{E}[\tau_{AB}]} $$

where $\rho_{AB}$ is the reactive probability and $\mathbb{E}[\tau_{AB}]$ denotes the mean transition time from $A$ to $B$. The quantity $\mathbb{E}[\tau_{AB}]$ can be sampled from the optimal controlled process and $\rho_{AB}$ depends on the committor.

The forward committor function $q^+$ satisfies the boundary value problem:

$$ \begin{cases}
b \cdot \nabla q^+ + \frac{1}{2} \sigma \sigma^T : \nabla \nabla q^+  = 0 & x \in \Omega \backslash (A \cup B)\\
q^+(x) = 0 & x \in \partial A\\
q^+(x) = 1 & x \in \partial B.
\end{cases} $$

Committor functions are solved using NN-based solvers that are adapted from methods in the following papers:

Qianxiao Li, Bo Lin, and Weiqing Ren, [Computing committor functions for the study of rare events using deep learning](https://aip.scitation.org/doi/10.1063/1.5110439). In: The Journal of Chemical Physics 151 (2019)

M. Raissi, P. Perdikaris, and G.E. Karniadakis. [Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial dif-
ferential equations](https://doi.org/https://doi.org/10.1016/j.jcp.2018.10.045). In: Journal of Computational Physics 378 (2019).

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu 5, Paris Perdikaris, Sifan Wang, and Liu
Yang. [Physics-informed machine Learning](https://www.nature.
com/articles/s42254-021-00314-5). In: Nature (2021).

FEM are also used as a comparison for the committor functions. For more details of the implementation, see https://github.com/mar1akc/transition_path_theory_FEM_distmesh

## Requirements:
- python 3.7
- torch 1.4.1

## Usage

### Mueller potential in 2D
- Train NN-solver for committor function: ./Mueller/mueller_metadynamics_uniform_in_space.ipynb
- Solve committor function through FEM: ./Mueller/FEM_TPT.py
- Compare results of committor using NN-solver and FEM: ./Mueller/evaluation_file.ipynb
- Compute transition rate: ./Mueller/transition rate- optimal control.ipynb

### Lennard-Jones-7 in 2D
- Train NN-solver for committor function: ./LJ7_2D/LJ7 - NN train.ipynb
- Solve committor function through FEM: ./LJ7_2D/LJ7/FEM_TPT.py
- Compute transition rate with controlled dynamics: ./LJ7_2D/LJ7-optimal control.ipynb
- Compute transition rate with unbiased dynamics in 14D: ./LJ7_2D/MALA transition rate.py

### Duffing Oscillator in 1D
- Train PINN-solver for committor function: ./Duffing/PINN_test.ipynb
- Solve committor function through FEM: ./Duffing/Duffing_committor_ellipse.ipynb
- Compare results of committor using NN-solver and FEM: ./Duffing/evaluation_file_PINN.ipynb
- Compute transition rate with controlled dynamics: ./Duffing/Estimating transition rate via sampling - optimal control
- Compute transition rate with unbiased dynamics: ./Duffing/Transition rate_unbiased.py

