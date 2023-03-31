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

$$ dY_t = (b(Y_t) + \sigma(Y_t) \sigma^T(Y_t) v(Y_t)) dt + \sigma(Y_t) dW_t $$


## Requirements:
- python 3.7
- torch 1.4.1

