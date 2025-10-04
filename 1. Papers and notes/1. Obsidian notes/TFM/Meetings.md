## Meeting 1 - Friday 19/09/2025
#### Milestone 1: 1.5 Months
- Get access to the server and play with Javier's code
- HDG discretization + [[Stokes Equation]]
- GPU algorithms
- HDG code in Matlab and Python
#### Further milestones:
1. GPU Bottleneck with memory transfer -> Dig into it
2. GPU HDG - Stokes Problem
3. GPU HDG - Linear Navier-Stokes

#### [[HDG Method]] Overview
- New unknown: Flux
- In integral form, flux can be discontinuous across elements (No derivative constraint, as in strong form)
- Flip faces due to the direction of the local nodes
- Lambda: Face unknown.    u, q: elemental unknowns
- f = -2 for error analysis (Analytical solution is u = x^2)
- If degree >= 2: zero error due to polynomial approximation (u = x^2)
- Cheap postprocess to reach p + 2 convergence

## Meeting 2 - Friday 03/10/2025

Solve [[Questions#Meeting 1]]
#### Next Steps
- Download [[Netgen-NGSolve]] 
	- Try the HDG - DG Methods
	- Compare the performance with our code
- Take a look on Xavi Roca's notes on sparse matrix-vector products to improve performance of the GPU code

