import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from referenceElement.evaluateNodalBasisTri import evaluate_nodal_basis_tri
from referenceElement.evaluateNodalBasisTriWithoutDerivatives import evaluate_nodal_basis_tri_without_derivatives

def plot_postprocessed_solution(X, T, u, reference_element_star, n_deg_ref=20):
    """
    Plots the postprocessed HDG solution (u_star) over the mesh using nodal interpolation.
    Rendered as a 2D colored surface (contour) plot.
    """

    # Generate equally spaced points in reference triangle
    nodes_ref = []
    h = 1 / n_deg_ref
    for j in range(n_deg_ref + 1):
        for i in range(n_deg_ref + 1 - j):
            nodes_ref.append([i * h, j * h])
    nodes_ref = np.array(nodes_ref)
    nodes_ref = 2 * nodes_ref - 1  # Map to [-1, 1] reference triangle

    # Triangulate reference triangle
    tri_ref = Delaunay(nodes_ref).simplices

    coord_ref = reference_element_star["NodesCoord"]
    coord_geo = reference_element_star["NodesCoordGeo"]
    degree = reference_element_star["degree"]
    n_nodes_per_elem = coord_ref.shape[0]
    n_elements = T.shape[0]

    # Evaluate basis functions for u_star and geometry
    N = evaluate_nodal_basis_tri_without_derivatives(nodes_ref, coord_ref, degree)
    N_geo = evaluate_nodal_basis_tri_without_derivatives(nodes_ref, coord_geo, degree - 1)

    x_all = []
    y_all = []
    u_all = []
    tri_global = []

    for ielem in range(n_elements):
        Xe = X[T[ielem, :], :]
        u_local = u[ielem * n_nodes_per_elem : (ielem + 1) * n_nodes_per_elem]

        Xg = N_geo @ Xe      # Geometry interpolated
        ug = N @ u_local     # Postprocessed u* interpolated

        idx_start = len(x_all)
        x_all.extend(Xg[:, 0])
        y_all.extend(Xg[:, 1])
        u_all.extend(ug)
        tri_global.extend((tri_ref + idx_start).tolist())

    x_all = np.array(x_all)
    y_all = np.array(y_all)
    u_all = np.array(u_all)
    tri_global = np.array(tri_global)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5))
    triang = Triangulation(x_all, y_all, tri_global)
    tpc = ax.tripcolor(triang, u_all, shading='gouraud', cmap='jet')
    ax.set_aspect('equal')
    ax.set_title("Postprocessed HDG solution: $u^*$")

    return fig, ax, tpc

