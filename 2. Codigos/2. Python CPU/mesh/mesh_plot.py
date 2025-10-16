import numpy as np
import matplotlib.pyplot as plt

def faceNodes_aux(nOfElementNodes):
   
    if nOfElementNodes == 3:  # P1 
        return [[0, 1], [1, 2], [2, 0]]
    elif nOfElementNodes == 6:  # P2 
        return [[0, 3, 1], [1, 4, 2], [2, 5, 0]]
    elif nOfElementNodes == 10:  # P3 
        return [[0, 3, 4, 1], [1, 5, 6, 2], [2, 7, 8, 0]]
    elif nOfElementNodes == 15:  # P4
        return [[0, 3, 4, 5, 1], [1, 6, 7, 8, 2], [2, 9, 10, 11, 0]]
    elif nOfElementNodes == 21:  # P5
        return [[0, 3, 4, 5, 6, 1], [1, 7, 8, 9, 10, 2], [2, 11, 12, 13, 14, 0]]
    elif nOfElementNodes == 28:  # P6
        return [[0, 3, 4, 5, 6, 7, 1], [1, 8, 9, 10, 11, 12, 2], [2, 13, 14, 15, 16, 17, 0]]
    elif nOfElementNodes == 36:  # P7
        return [[0, 3, 4, 5, 6, 7, 8, 1], [1, 9, 10, 11, 12, 13, 14, 2], [2, 15, 16, 17, 18, 19, 20, 0]]
    elif nOfElementNodes == 45:  # P8
        return [[0, 3, 4, 5, 6, 7, 8, 9, 1], [1, 10, 11, 12, 13, 14, 15, 16, 2], [2, 17, 18, 19, 20, 21, 22, 23, 0]]
    elif nOfElementNodes == 55:  # P9
        return [[0, 3, 4, 5, 6, 7, 8, 9, 10, 1], [1, 11, 12, 13, 14, 15, 16, 17, 18, 2], [2, 19, 20, 21, 22, 23, 24, 25, 26, 0]]
    elif nOfElementNodes == 66:  # P10
        return [[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1], [1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2], [2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0]]
    elif nOfElementNodes == 78:  # P11
        return [[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1], [1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 2], [2, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0]]
    else:
        raise ValueError("Non valid number. Supported values: 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78.")

def plot_mesh(X, T, option=".", nodesNum="."):
 
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    for element in T:
        nNodesElem = len(element)
        faceConn = faceNodes_aux(nNodesElem)
        for face in faceConn:
            face_nodes = element[face]
            coords = X[face_nodes, :]  
            ax.plot(coords[:, 0], coords[:, 1], 'k-', lw=1)
    
    # opt
    if option == 'plotNodes':
        ax.scatter(X[:, 0], X[:, 1], color='blue', s=10, zorder=3)
    elif option == 'plotNodesNum':
        if nodesNum == 'all':
            nodesNum = range(X.shape[0])
        for inode in nodesNum:
            ax.annotate(str(inode), (X[inode, 0], X[inode, 1]), fontsize=10, color='red')
    elif option == 'plotElements':
        for iElem, element in enumerate(T):
            centroid = np.mean(X[element], axis=0)
            ax.annotate(str(iElem), centroid, fontsize=12, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('FE Mesh')
    plt.show()


