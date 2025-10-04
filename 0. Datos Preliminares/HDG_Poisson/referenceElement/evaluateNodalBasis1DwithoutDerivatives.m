function N=evaluateNodalBasis1DwithoutDerivatives(Xi,XiNodes,degree)

[V,kk]=orthogonalPolynomialsAndDerivatives1D(degree,XiNodes);

[P,dPdxi]=orthogonalPolynomialsAndDerivatives1D(degree,Xi);

N = P/V;

