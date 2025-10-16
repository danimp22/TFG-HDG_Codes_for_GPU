import numpy as np
from scipy.sparse import diags
from numpy.linalg import solve

import poisson.analyticalPoisson
from poisson.analyticalPoisson import analytical_poisson

def compute_projection_faces(data_function, faces, X, T, reference_element):
    """
    Projects the given function onto the face basis functions using L2 projection.

    Parameters
    ----------
    data_function : callable
        Function u(x, y) to be projected. Takes array of shape (N, 2) and returns (N,) or (N, nComp).
    faces : ndarray
        Array of shape (nFaces, 2) where each row = [element_index, local_face_index] 
        (ya en 0-indexation).
    X : ndarray
        Node coordinates, shape (nNodes, 2).
    T : ndarray
        Element connectivity, shape (nElements, nNodesPerElement) (ya en 0-indexation).
    reference_element : dict
        Must contain:
            - NodesCoord1d (nFaceNodes, 1)
            - faceNodes (nFacesPerElem, nFaceNodes) (ya en 0-indexation)
            - N1d (ngf, nFaceNodes)
            - N1dxi (ngf, nFaceNodes)
            - IPweights1d (ngf,)

    Returns
    -------
    u : ndarray
        Projected nodal values on faces, shape (nFaces * nFaceNodes * nComponents,)
    """

    n_faces = faces.shape[0]
    n_face_nodes = reference_element["NodesCoord1d"].shape[0]
    face_nodes = reference_element["faceNodes"]
    N1d = reference_element["N1d"]              # (ngf, n_face_nodes)
    Nxi1d = reference_element["N1dxi"]          # (ngf, n_face_nodes)
    IPw_f = reference_element["IPweights1d"]    # (ngf,)
    ngf = len(IPw_f)

    # Verify number of components
    sample_val = data_function(np.array([[1.0, 1.0]]))
    n_components = sample_val.shape[1] if sample_val.ndim == 2 else 1
    n_dof_face = n_face_nodes * n_components

    u = np.zeros(n_faces * n_dof_face)

    for f in range(n_faces):
        i_elem = faces[f, 0]
        i_face = faces[f, 1]
        Te = T[i_elem]
        Xe = X[Te]

        # Local nodes of faces
        nodes = face_nodes[i_face]
        xf = Xe[nodes, 0]
        yf = Xe[nodes, 1]

        # Evaluate Gauss points in faces
        xfg = N1d @ xf
        yfg = N1d @ yf
        xyg = np.column_stack((xfg, yfg))

        # Evaluate data function in gauss points
        ug = data_function(xyg)
        if n_components == 1:
            ug = ug.reshape(ngf, 1)  # shape(ngf, 1)

        # Jacobian computation
        dxdxi = Nxi1d @ xf
        dydxi = Nxi1d @ yf
        dxdxi_norm = np.sqrt(dxdxi**2 + dydxi**2)

        # Pondered integration matrix
        dline = diags(dxdxi_norm * IPw_f)
        M = N1d.T @ (dline @ N1d)  # (n_face_nodes, n_face_nodes)

        # Projection of each component
        for comp in range(n_components):
            b = N1d.T @ (dline @ ug[:, comp])
            uface = solve(M, b)
            offset = f * n_dof_face + comp * n_face_nodes
            u[offset : offset + n_face_nodes] = uface

    return u

