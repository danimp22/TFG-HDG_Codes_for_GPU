import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from referenceElement.evaluateNodalBasisTri import evaluate_nodal_basis_tri

def plot_discontinuous_solution(X, T, u, reference_element, n_deg_ref=20):
    """
    Plots the discontinuous solution from HDG using nodal interpolation.
    This version renders the result in 2D similar to MATLAB.
    """
    # Generar puntos igualmente espaciados en el triángulo de referencia
    nodes_ref = []
    h = 1 / n_deg_ref
    for j in range(n_deg_ref + 1):
        for i in range(n_deg_ref + 1 - j):
            nodes_ref.append([i * h, j * h])
    nodes_ref = np.array(nodes_ref)
    nodes_ref = 2 * nodes_ref - 1  # Mapeo a [-1,1]x[-1,1]

    # Triangulación del triángulo de referencia
    from scipy.spatial import Delaunay
    tri_ref = Delaunay(nodes_ref).simplices

    coord_ref = reference_element["NodesCoord"]
    degree = reference_element["degree"]
    n_nodes_per_elem = coord_ref.shape[0]
    n_elements = T.shape[0]

    # Evaluar funciones de forma en los puntos de interpolación
    from referenceElement.evaluateNodalBasisTri import evaluate_nodal_basis_tri
    N, _, _ = evaluate_nodal_basis_tri(nodes_ref, coord_ref, degree)

    x_all = []
    y_all = []
    u_all = []
    tri_global = []

    for ielem in range(n_elements):
        Te = T[ielem, :]
        Xe = X[Te, :]
        Xg = N @ Xe  # Coordenadas interpoladas
        ulocal = u[ielem * n_nodes_per_elem:(ielem + 1) * n_nodes_per_elem]
        ug = N @ ulocal  # Valores interpolados

        idx_start = len(x_all)
        x_all.extend(Xg[:, 0])
        y_all.extend(Xg[:, 1])
        u_all.extend(ug)
        tri_global.extend((tri_ref + idx_start).tolist())

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    u_all = np.array(u_all)
    tri_global = np.array(tri_global)

    # Creación de la figura y ejes
    fig, ax = plt.subplots(figsize=(6, 5))
    from matplotlib.tri import Triangulation
    triang = Triangulation(x_all, y_all, tri_global)
    tpc = ax.tripcolor(triang, u_all, shading='gouraud', cmap='jet')
    ax.set_aspect('equal')
    ax.set_title("HDG solution: u")
    
    # No llamamos a plt.show() aquí
    return fig, ax, tpc
