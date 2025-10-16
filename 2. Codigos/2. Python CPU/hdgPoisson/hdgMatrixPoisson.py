import numpy as np
from scipy.sparse import diags
from numpy.linalg import solve
from scipy.sparse import csr_matrix

import poisson.sourcePoisson
from poisson.sourcePoisson import sourcePoisson

def hdgMatrixPoisson(muElem, X, T, F, referenceElement, infoFaces, tau):

    
    nOfFaces = np.max(F)+1
    nOfElements = T.shape[0]
    nOfInteriorFaces = infoFaces['intFaces'].shape[0]
    nOfFaceNodes = referenceElement['NodesCoord1d'].shape[0]
    nDOF = nOfFaces * nOfFaceNodes
    f = np.zeros(nDOF)
    QQ = [None] * nOfElements
    UU = [None] * nOfElements
    Qf = [None] * nOfElements
    Uf = [None] * nOfElements

    n = nOfElements * (3 * nOfFaceNodes)**2
    ind_i = np.zeros(n, dtype=int)
    ind_j = np.zeros(n, dtype=int)
    coef_K = np.zeros(n)
    indK = 0

    

    # loop in elements

    for iElem in range(nOfElements):
        Te = T[iElem]
        Xe = X[Te]
        Fe = F[iElem]
        isFeInterior = Fe < nOfInteriorFaces # Boolean (1=interior face)

        # elemental matrices
        Qe, Ue, Qfe, Ufe, Alq, Alu, All = KKeElementalMatricesIsoParametric(
            muElem[iElem], Xe, referenceElement, tau[iElem]
        )
        
        # Interior faces seen from the second element are flipped to have proper orientation
        flipFace = np.zeros(3, dtype=bool)
        for iface in range(3):
            if Fe[iface] < nOfInteriorFaces:

                face_info = infoFaces['intFaces'][Fe[iface]]
              
                elem_actual = iElem
                # Invert face if current element is greater than both neighbors
                # (only second element see inverted face)
                if elem_actual == max(face_info[0], face_info[2]):
                    flipFace[iface] = True
                else:
                    flipFace[iface] = False

        indL = np.arange(3 * nOfFaceNodes) # permutation for local numbering
        aux = np.arange(nOfFaceNodes-1, -1, -1)
        indflip = np.concatenate([aux, nOfFaceNodes + aux, 2 * nOfFaceNodes + aux])
        flip_vector = np.concatenate([
            np.ones(nOfFaceNodes) * flipFace[0],
            np.ones(nOfFaceNodes) * flipFace[1],
            np.ones(nOfFaceNodes) * flipFace[2]
        ])
        indL[flip_vector == 1] = indflip[flip_vector == 1]

        # reorder by orientation
        Qe = Qe[:, indL]
        Ue = Ue[:, indL]
        Alq = Alq[indL, :]
        Alu = Alu[indL, :]
        All = All[np.ix_(indL, indL)]

        # Save local maps for postprocessing
        QQ[iElem] = csr_matrix(Qe)
        UU[iElem] = csr_matrix(Ue)
        Qf[iElem] = Qfe
        Uf[iElem] = Ufe

       
        KKe = (Alq @ Qe + Alu @ Ue + All).T
        ffe = -(Alq @ Qfe + Alu @ Ufe)
        """print(f"Kloc elemento {iElem}:\n", KKe)
        print(f"floc elemento {iElem}:\n", ffe)
      """
        aux = np.arange(nOfFaceNodes)
        indRC = np.concatenate([
            Fe[0] * nOfFaceNodes + aux,
            Fe[1] * nOfFaceNodes + aux,
            Fe[2] * nOfFaceNodes + aux
        ]).astype(int)


        f[indRC] += ffe.flatten()

        
        for irow in range(3 * nOfFaceNodes):
            for icol in range(3 * nOfFaceNodes):
                ind_i[indK] = indRC[irow]
                ind_j[indK] = indRC[icol]
                coef_K[indK] = KKe[irow, icol]
                indK += 1
    
    KK = csr_matrix((coef_K, (ind_i, ind_j)), shape=(nDOF, nDOF))

    return KK, f, QQ, UU, Qf, Uf

    

def KKeElementalMatricesIsoParametric(mu, Xe, referenceElement, tau):
    
    nOfElementNodes = referenceElement['NodesCoord'].shape[0]
    nOfFaceNodes = referenceElement['NodesCoord1d'].shape[0]
    faceNodes = referenceElement['faceNodes']
    nOfFaces = 3

    N = referenceElement['N']
    Nxi = referenceElement['Nxi']
    Neta = referenceElement['Neta']
    N1d = referenceElement['N1d']
    Nx1d = referenceElement['N1dxi']
    IPw_f = referenceElement['IPweights1d']
    ngf = len(IPw_f)
    IPw = referenceElement['IPweights']
    ngauss = len(IPw)

    # Jacobian and derivatives
    J11 = Nxi @ Xe[:, 0]
    J12 = Nxi @ Xe[:, 1]
    J21 = Neta @ Xe[:, 0]
    J22 = Neta @ Xe[:, 1]
    detJ = J11 * J22 - J12 * J21

    dvolu = diags(IPw * detJ)
    invJ11 = diags(J22 / detJ)
    invJ12 = diags(-J12 / detJ)
    invJ21 = diags(-J21 / detJ)
    invJ22 = diags(J11 / detJ)

    Nx = invJ11 @ Nxi + invJ12 @ Neta
    Ny = invJ21 @ Nxi + invJ22 @ Neta

    Xg = N @ Xe
    sourceTerm = sourcePoisson(Xg, mu)  
    fe = N.T @ (dvolu @ sourceTerm)

    Me = N.T @ (dvolu @ N)
    Aqq = np.zeros((2 * nOfElementNodes, 2 * nOfElementNodes))
    aux = np.arange(0, 2 * nOfElementNodes, 2)
    aux2 = aux + 1
    Aqq[np.ix_(aux, aux)] = Me
    Aqq[np.ix_(aux2, aux2)] = Me

    Auq = np.zeros((nOfElementNodes, 2 * nOfElementNodes))
    Auq[:, aux] = N.T @ (dvolu @ Nx)
    Auq[:, aux2] = N.T @ (dvolu @ Ny)

    Alq = np.zeros((3 * nOfFaceNodes, 2 * nOfElementNodes))
    Auu = np.zeros((nOfElementNodes, nOfElementNodes))
    Alu = np.zeros((3 * nOfFaceNodes, nOfElementNodes))
    All = np.zeros((3 * nOfFaceNodes, 3 * nOfFaceNodes))

    for iface in range(nOfFaces):
        tau_f = tau[iface]
        nodes = faceNodes[iface] 
        Xf = Xe[nodes]
        dxdxi = Nx1d @ Xf[:, 0]
        dydxi = Nx1d @ Xf[:, 1]
        dxdxiNorm = np.sqrt(dxdxi**2 + dydxi**2)
        dline = dxdxiNorm * IPw_f
        tx = dxdxi
        ty = dydxi
        norm = np.sqrt(tx**2 + ty**2)
        nx = ty / norm
        ny = -tx / norm

        ind_face = iface * nOfFaceNodes + np.arange(nOfFaceNodes)
        for i, node in enumerate(nodes):
            Alq[ind_face, 2 * node    ] += (N1d.T @ (diags(dline * nx) @ N1d))[:, i]
            Alq[ind_face, 2 * node + 1] += (N1d.T @ (diags(dline * ny) @ N1d))[:, i]


        Auu_f = N1d.T @ (diags(dline) @ N1d) * tau_f
        Auu[np.ix_(nodes, nodes)] += Auu_f
        Alu[np.ix_(ind_face, nodes)] += Auu_f
        All[np.ix_(ind_face, ind_face)] = -Auu_f

    Aqu = -mu * Auq.T
    Aul = -Alu.T
    Aql = mu * Alq.T
    A = np.block([[Auu, Auq], [Aqu, Aqq]])

    UQ = -solve(A, np.vstack((Aul, Aql)))
    fUQ = solve(A, np.vstack((fe.reshape(-1, 1), np.zeros((2 * nOfElementNodes, 1)))))

    U = UQ[:nOfElementNodes, :]
    Uf = fUQ[:nOfElementNodes].flatten()
    Q = UQ[nOfElementNodes:, :]
    Qf = fUQ[nOfElementNodes:].flatten()

    return Q, U, Qf, Uf, Alq, Alu, All
