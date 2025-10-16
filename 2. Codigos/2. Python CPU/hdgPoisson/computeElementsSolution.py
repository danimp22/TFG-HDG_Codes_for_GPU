import numpy as np

def compute_elements_solution(lambda_, UU, QQ, Uf, Qf, F):
    n_elements = F.shape[0]
    n_element_nodes, aux = UU[0].shape
    n_face_nodes = aux // 3

    u = np.zeros(n_elements * n_element_nodes)
    q = np.zeros(2 * n_elements * n_element_nodes)

    for ielem in range(n_elements):
        Fe = F[ielem, :]  
        aux = np.arange(n_face_nodes)
        # Compute global indices of lambda
        ind = np.concatenate([Fe[i] * n_face_nodes + aux for i in range(3)])

        u_elem = UU[ielem] @ lambda_[ind] + Uf[ielem]
        q_elem = QQ[ielem] @ lambda_[ind] + Qf[ielem]

        u[ielem * n_element_nodes : (ielem + 1) * n_element_nodes] = u_elem
        q[ielem * 2 * n_element_nodes : (ielem + 1) * 2 * n_element_nodes] = q_elem

    # Reorder q with shape: (n_elements * n_element_nodes, 2)
    q = q.reshape((n_elements * n_element_nodes, 2))

    return u, q

