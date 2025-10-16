import numpy as np

def compute_L2_norm(reference_element, X, T, u, u0, *args):
    """
    Computes the L2 Norm of the error between FEM solution and analytical function.

    Parameters
    ----------
    reference_element : object
        Reference element containing IPweights, N, Nxi, Neta.
    X : ndarray
        Nodal coordinates (nNodes x 2).
    T : ndarray
        Connectivity matrix (nElements x nNodesPerElem).
    u : ndarray
        FEM solution at nodes (flattened element-wise).
    u0 : function
        Analytical function of the form u0(xy, *args).
    *args : list
        Additional arguments passed to the analytical function.

    Returns
    -------
    float
        L2 norm of the error.
    """
    n_elements, n_nodes_per_elem = T.shape
    L2Norm = 0.0

    for i_elem in range(n_elements):
        Te = T[i_elem, :]
        Xe = X[Te, :]
        ind = slice(i_elem * n_nodes_per_elem, (i_elem + 1) * n_nodes_per_elem)
        ue = u[ind]
        L2Norm += elemental_L2_norm(Xe, reference_element, ue, u0, *args)

    return np.sqrt(L2Norm)


def elemental_L2_norm(Xe, reference_element, ue, u0, *args):
    IPw = reference_element["IPweights"]
    N = reference_element["N"]
    Nxi = reference_element["Nxi"]
    Neta = reference_element["Neta"]

    xe = Xe[:, 0]
    ye = Xe[:, 1]
    elem_L2_norm = 0.0

    for g in range(len(IPw)):
        N_g = N[g, :]
        Nxi_g = Nxi[g, :]
        Neta_g = Neta[g, :]

        xy_g = N_g @ Xe
        ue_g = N_g @ ue
        u0_g = u0(np.array([xy_g]), *args)[0]

        J = np.array([
            [Nxi_g @ xe, Nxi_g @ ye],
            [Neta_g @ xe, Neta_g @ ye]
        ])

        dvolu = IPw[g] * np.linalg.det(J)
        elem_L2_norm += (ue_g - u0_g) ** 2 * dvolu

    return elem_L2_norm
