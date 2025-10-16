import numpy as np

from referenceElement.evaluateNodalBasisTriWithoutDerivatives import evaluate_nodal_basis_tri_without_derivatives

def HDG_postprocess(X, T, u, q, reference_element_star):
    """
    Versión CPU del postproceso HDG que imprime debug sólo para el elemento 0.
    """
    n_elements      = T.shape[0]
    n_element_nodes = T.shape[1]
    coord_ref_star  = reference_element_star['NodesCoord']
    npoints         = coord_ref_star.shape[0]

    # 1) Shape functions enriquecidas
    degree_k = reference_element_star['degree'] - 1
    shape_functions = evaluate_nodal_basis_tri_without_derivatives(
        reference_element_star['NodesCoord'],
        reference_element_star['NodesCoordGeo'],
        degree_k
    )
    # (Sólo imprimirlas una vez)
    print("Shape functions evaluated at star nodes:\n", shape_functions)

    u_star = np.zeros(npoints * n_elements)

    for i_elem in range(n_elements):
        ind      = slice(i_elem * n_element_nodes, (i_elem + 1) * n_element_nodes)
        ind_star = slice(i_elem * npoints,      (i_elem + 1) * npoints)

        ue = shape_functions @ u[ind]
        qe = shape_functions @ q[ind, :]

        if i_elem == 0:
            print(f"\nCPU – elemento 0:")
            print("  ue =", ue)
            print("  qe =\n", qe)
            print("\nReference element star properties (elemento 0):")
            print("  IP weights:", reference_element_star['IPweights'])
            print("  Shape functions N:\n", reference_element_star['N'])
            print("  Shape derivatives Nxi:\n", reference_element_star['Nxi'])
            print("  Shape derivatives Neta:\n", reference_element_star['Neta'])
            print("  Geometric dNGeodxi:\n", reference_element_star['dNGeodxi'])
            print("  Geometric dNGeodeta:\n", reference_element_star['dNGeodeta'])

        Ke, Bq, int_u_star, int_u = elemental_matrices(
            X[T[i_elem, :], :],
            reference_element_star,
            ue, qe,
            debug=(i_elem == 0)
        )

        # montar y resolver sistema de Lagrange
        K_block = np.block([
            [Ke,               int_u_star.reshape(-1,1)],
            [int_u_star.reshape(1,-1),        np.zeros((1,1))]
        ])
        f = np.concatenate([Bq, [int_u]])
        sol = np.linalg.solve(K_block, f)
        u_star[ind_star] = sol[:-1]

    return u_star


def elemental_matrices(Xe, reference_element_star, ue, qe, debug=False):
    """
    Monta Ke, Bq, int_u_star e int_u para un elemento.
    Si debug=True, imprime J, Nx_g y Ny_g en cada punto de Gauss.
    """
    n_nodes_star = reference_element_star['NodesCoord'].shape[0]
    K            = np.zeros((n_nodes_star, n_nodes_star))
    Bq           = np.zeros(n_nodes_star)
    int_u_star   = np.zeros(n_nodes_star)
    int_u        = 0.0

    IPw     = reference_element_star['IPweights']
    N       = reference_element_star['N']
    Nxi     = reference_element_star['Nxi']
    Neta    = reference_element_star['Neta']
    NxiGeo  = reference_element_star['dNGeodxi']
    NetaGeo = reference_element_star['dNGeodeta']

    xe = Xe[:, 0]
    ye = Xe[:, 1]

    for g in range(len(IPw)):
        N_g     = N[g, :]
        Nxi_g   = Nxi[g, :]
        Neta_g  = Neta[g, :]

        # Jacobiano
        J = np.array([
            [NxiGeo[g]  @ xe,   NxiGeo[g]  @ ye],
            [NetaGeo[g] @ xe,   NetaGeo[g] @ ye]
        ])
        dvolu = IPw[g] * np.linalg.det(J)
        invJ  = np.linalg.inv(J)

        # derivadas físicas de N
        Nx_g = invJ[0,0] * Nxi_g + invJ[0,1] * Neta_g
        Ny_g = invJ[1,0] * Nxi_g + invJ[1,1] * Neta_g

        if debug:
            print(f"  g={g}: J =\n{J}")
            print(f"       Nx = {Nx_g}")
            print(f"       Ny = {Ny_g}")

        # valores en Gauss
        u_g  = N_g @ ue
        qx_g = N_g @ qe[:,0]
        qy_g = N_g @ qe[:,1]

        # ensamblaje elemental
        Bq         += (Nx_g * qx_g + Ny_g * qy_g) * dvolu
        K          += (np.outer(Nx_g, Nx_g) + np.outer(Ny_g, Ny_g)) * dvolu
        int_u_star += N_g * dvolu
        int_u     += u_g * dvolu

    return K, Bq, int_u_star, int_u
