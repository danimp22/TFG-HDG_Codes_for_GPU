import numpy as np

from referenceElement.evaluateNodalBasisTriWithoutDerivatives import evaluate_nodal_basis_tri_without_derivatives

def HDG_postprocess(X, T, u, q, reference_element_star):
    """
    Postprocess HDG solution to obtain an enhanced scalar field u_star.

    Parameters:
    - X: (nNodes x 2) array of node coordinates
    - T: (nElements x nNodesPerElement) array of element connectivity
    - u: global solution vector
    - q: global gradient matrix (2 columns: [qx, qy])
    - reference_element_star: enriched reference element

    Returns:
    - u_star: enhanced scalar field (per element, per enriched node)
    """
    n_elements = T.shape[0]
    n_element_nodes = T.shape[1]
    coord_ref_star = reference_element_star['NodesCoord']
    npoints = coord_ref_star.shape[0]

    # Compute shape functions of degree k at star nodes
    degree_k = reference_element_star['degree'] - 1
    shape_functions = evaluate_nodal_basis_tri_without_derivatives(
        reference_element_star['NodesCoord'],
        reference_element_star['NodesCoordGeo'],
        degree_k
    )
    """print("Shape functions evaluated at star nodes:\n", shape_functions)"""
    u_star = np.zeros(npoints * n_elements)

    for i_elem in range(n_elements):
        ind = slice(i_elem * n_element_nodes, (i_elem + 1) * n_element_nodes)
        ind_star = slice(i_elem * npoints, (i_elem + 1) * npoints)

        ue = shape_functions @ u[ind]
        qe = shape_functions @ q[ind, :]  # shape (npoints, 2)

        """print(f"Element {i_elem}: ue = {ue}, qe = {qe}")"""

        Ke, Bq, int_u_star, int_u = elemental_matrices(
            X[T[i_elem, :], :],
            reference_element_star,
            ue,
            qe
        )

        if i_elem == 0:
            IPw = reference_element_star['IPweights']
            N = reference_element_star['N']
            Nxi = reference_element_star['Nxi']
            Neta = reference_element_star['Neta']
            NxiGeo = reference_element_star['dNGeodxi']
            NetaGeo = reference_element_star['dNGeodeta']

            """print("Reference element star properties:")
            print("IP weights:", IPw)  
            print("Shape functions N:", N)
            print("Shape function derivatives Nxi:", Nxi)   
            print("Shape function derivatives Neta:", Neta) 
            print("Geometric shape function derivatives NxiGeo:", NxiGeo)  
            print("Geometric shape function derivatives NetaGeo:", NetaGeo) """


        # Lagrange multiplier system
        K = np.block([
            [Ke, int_u_star.reshape(-1, 1)],
            [int_u_star.reshape(1, -1), np.zeros((1, 1))]
        ])
        f = np.concatenate([Bq, [int_u]])

        sol = np.linalg.solve(K, f)
        u_star[ind_star] = sol[:-1]

    return u_star

def elemental_matrices(Xe, reference_element_star, ue, qe):
    n_nodes_star = reference_element_star['NodesCoord'].shape[0]
    K = np.zeros((n_nodes_star, n_nodes_star))
    Bq = np.zeros(n_nodes_star)
    int_u_star = np.zeros(n_nodes_star)
    int_u = 0.0

    IPw = reference_element_star['IPweights']
    N = reference_element_star['N']
    Nxi = reference_element_star['Nxi']
    Neta = reference_element_star['Neta']
    NxiGeo = reference_element_star['dNGeodxi']
    NetaGeo = reference_element_star['dNGeodeta']

    """
    print("Reference element star properties:")
    print("IP weights:", IPw)  
    print("Shape functions N:", N)
    print("Shape function derivatives Nxi:", Nxi)   
    print("Shape function derivatives Neta:", Neta) 
    print("Geometric shape function derivatives NxiGeo:", NxiGeo)  
    print("Geometric shape function derivatives NetaGeo:", NetaGeo) """

    xe = Xe[:, 0]
    ye = Xe[:, 1]

    ngauss = len(IPw)

    for g in range(ngauss):
        N_g = N[g, :]
        Nxi_g = Nxi[g, :]
        Neta_g = Neta[g, :]

        J = np.array([
            [NxiGeo[g, :] @ xe, NxiGeo[g, :] @ ye],
            [NetaGeo[g, :] @ xe, NetaGeo[g, :] @ ye]
        ])
        dvolu = IPw[g] * np.linalg.det(J)
        invJ = np.linalg.inv(J)

        Nx_g = invJ[0, 0] * Nxi_g + invJ[0, 1] * Neta_g
        Ny_g = invJ[1, 0] * Nxi_g + invJ[1, 1] * Neta_g

        u_g = N_g @ ue
        qx_g = N_g @ qe[:, 0]
        qy_g = N_g @ qe[:, 1]

        Bq += (Nx_g * qx_g + Ny_g * qy_g) * dvolu
        K += (np.outer(Nx_g, Nx_g) + np.outer(Ny_g, Ny_g)) * dvolu
        int_u_star += N_g * dvolu
        int_u += u_g * dvolu
        """print(f"Gauss point {g}: dvolu = {dvolu}, u_g = {u_g}, qx_g = {qx_g}, qy_g = {qy_g}")
        print(f"Elemental matrices at Gauss point {g}: K = {K}, Bq = {Bq}, int_u_star = {int_u_star}, int_u = {int_u}")
        print(f"Jacobian at Gauss point {g}: J = {J}, invJ = {invJ}")"""

    return K, Bq, int_u_star, int_u
