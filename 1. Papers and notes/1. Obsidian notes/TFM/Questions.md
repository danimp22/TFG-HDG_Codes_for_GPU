## Meeting 1

In Javier's TFM we use a ref. triangle with points ((1, 0), (0, 1), (0, 0)) and in the code ((-1, -1), (1, -1), (-1, 1))?
- Follow the triangle in the code, the TFM is wrong

Why shape functions have 3 columns? Physical meaning behind.
- These are the values of the 3 shape functions (for 3 nodes in an element) at each of the Gauss Points

Why 7 Integration Points when degree = 1?
- Enough amount of IP while maintaining performance. IP errors are similar to other errors' magnitude.
- There are pre-established number of nodes for each degree of the polynomial (p = 1, Ngauss = 7; p = 2, Ngauss = 15)

Literature on discretizing the HDG.
- Bernardo Cockburn - Stokes Flow
- Bernardo Cockburn - A superconvergent LDG-hybridizable

Inefficient use of sparse matrices instead of loops?
- Javier didn't use them, check his code

In Dirichlet BC, why not apply them directly to the points. Also, negative values with solution u = x^2.
- For complex solutions, it gives less error when applying the conditions in the weak form rather than in the strong form.
- The negative values is because of the polynomial approximation of the faces (p = 1) while the solution is of order 2 (u = x^2)

## Meeting 2
