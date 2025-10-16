---
Date: 2025-09-19
---
## Milestone 1: 1.5 Months
- Get access to the server and play with Javier's code
- HDG discretization + [[Stokes Equation]]
- GPU algorithms
- HDG code in Matlab and Python
## Further milestones:
1. GPU Bottleneck with memory transfer -> Dig into it
2. GPU HDG - Stokes Problem
3. GPU HDG - Linear Navier-Stokes

## [[HDG Method]] Overview
- New unknown: Flux
- In integral form, flux can be discontinuous across elements (No derivative constraint, as in strong form)
- Flip faces due to the direction of the local nodes
- Lambda: Face unknown.    u, q: elemental unknowns
- $f = -2$ for error analysis (Analytical solution is $u = x^2$)
- If $degree >= 2$: zero error due to polynomial approximation ($u = x^2$)
- Cheap postprocess to reach $p + 2$ convergence