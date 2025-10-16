import numpy as np

def compute_L2_norm_postprocess(reference_element_star, X, T, u, u0, *args):
    """
    Computes the L2 Norm of the error between postprocessed FEM solution and analytical function.

    Parameters
    ----------
    reference_element_star : dict
        Dictionary with keys: 'IPweights', 'N', 'NGeo', 'dNGeodxi', 'dNGeodeta', 'NodesCoord'
    X : ndarray
        Nodal coordinates (nNodes x 2).
    T : ndarray
        Connectivity matrix (nElements x nNodesPerElem).
    u : ndarray
        Postprocessed FEM solution (flattened element-wise).
    u0 : function
        Analytical solution of the form u0([[x, y], ...], *args).
    *args : list
        Additional arguments passed to the analytical function.

    Returns
    -------
    float
        L2 norm of the error.
    """
    n_elements = T.shape[0]
    n_nodes_per_elem = reference_element_star["NodesCoord"].shape[0]

    L2Norm = 0.0
    for i_elem in range(n_elements):
        Te = T[i_elem, :]
        Xe = X[Te, :]
        ind = slice(i_elem * n_nodes_per_elem, (i_elem + 1) * n_nodes_per_elem)
        ue = u[ind]
        L2Norm += elemental_L2_norm_post(Xe, reference_element_star, ue, u0, *args)

    return np.sqrt(L2Norm)

def elemental_L2_norm_post(Xe, reference_element, ue, u0, *args):
    IPw = reference_element["IPweights"]
    N = reference_element["N"]
    NGeo = reference_element["NGeo"]
    NxiGeo = reference_element["dNGeodxi"]
    NetaGeo = reference_element["dNGeodeta"]

    xe = Xe[:, 0]
    ye = Xe[:, 1]

    elem_L2_norm = 0.0
    for g in range(len(IPw)):
        N_g = N[g, :]
        Nxi_g = NxiGeo[g, :]
        Neta_g = NetaGeo[g, :]
        xy_g = NGeo[g, :] @ Xe
        ue_g = N_g @ ue
        u0_g = u0(np.array([xy_g]), *args)[0]  # Pass 2D point to u0

        J = np.array([
            [Nxi_g @ xe, Nxi_g @ ye],
            [Neta_g @ xe, Neta_g @ ye]
        ])

        dvolu = IPw[g] * np.linalg.det(J)
        elem_L2_norm += (ue_g - u0_g) ** 2 * dvolu

    return elem_L2_norm
