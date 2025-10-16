import numpy as np

from referenceElement.orthogonalPolynomialsTri import orthogonal_polynomials_tri
from referenceElement.orthogonalPolynomialsAndDerivativesTri import orthogonal_polynomials_and_derivatives_tri

def evaluate_nodal_basis_tri(XI, XInodes, degree):
    """
    Evaluates nodal basis of polynomials for a given degree and nodes in XInodes.
    
    Parameters:
    XI (ndarray): Evaluation points (N,2).
    XInodes (ndarray): Nodes where the basis is defined (M,2).
    degree (int): Polynomial degree.
    
    Returns:
    N (ndarray): Evaluated shape functions.
    dNdxi (ndarray): Derivatives w.r.t xi.
    dNdeta (ndarray): Derivatives w.r.t eta.
    """
    V = orthogonal_polynomials_tri(degree, XInodes)
    invV = np.linalg.inv(V)
    
    P, dPdxi, dPdeta = orthogonal_polynomials_and_derivatives_tri(degree, XI)#, None, None  # Assuming derivative computation is separate
    N = np.dot(P, invV)
    Nxi = np.dot(dPdxi, invV)
    Neta = np.dot(dPdeta, invV)

    #dNdxi, dNdeta = None, None  # Derivatives not yet implemented
    
    return N, Nxi, Neta