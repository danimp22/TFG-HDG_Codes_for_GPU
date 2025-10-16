import numpy as np
import os

def generate_python_mesh_file(directory=".", files=None, face_nodes=None, inner_nodes=None, face_nodes1d=None):
    """
    Converts .dcm files into a Python-compatible format (.npz).
    
    Parameters:
        directory (str): Folder where the .dcm files are located (default is the current directory).
        files (list): List of .dcm files to process. If None, processes all files in `directory`.
        face_nodes (np.array): Face numbering matrix for 2D elements.
        inner_nodes (np.array): Array of internal nodes.
        face_nodes1d (np.array): Array of nodes for 1D elements.

    Returns:
        list: List of generated `.npz` files.
    """
    
    if files is None:
        files = [f for f in os.listdir(directory) if f.endswith(".dcm")]

    npz_files = []

    for dcm_file in files:
        file_path = os.path.join(directory, dcm_file)

        # Read the .dcm file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # First line: number of nodes, number of elements, element type
        n_nodes, n_elements, elem_type = map(int, lines[0].split())

        # Read nodes (index, x, y)
        X = np.array([list(map(float, line.split()[1:3])) for line in lines[1:n_nodes+1]])  # Solo toma X e Y

        # Read elements (connectivity) - Convert to 1-based indexing
        T = np.array([list(map(int, line.split()[7:])) for line in lines[n_nodes+1:n_nodes+1+n_elements]], dtype=int)

        # Read boundary conditions (Dirichlet and others)
        Tb = {}
        for line in lines[n_nodes+1+n_elements:]:
            if line.strip():  # Ignore empty lines
                parts = line.split()
                key = "Tb_Dirichlet" if parts[-1] == "Dirichlet" else "Tb_Other"
                if key not in Tb:
                    Tb[key] = []
                Tb[key].append(list(map(int, parts[:2])))  # Tomamos solo los dos primeros valores

        for key in Tb:
            Tb[key] = np.array(Tb[key])

        # Apply custom numbering if provided
        if face_nodes is not None and inner_nodes is not None:
            order = np.concatenate([face_nodes.flatten(), inner_nodes.flatten()])
            T = T[:, order]

        if face_nodes1d is not None:
            for key in Tb:
                Tb[key] = Tb[key][:, face_nodes1d]

        # Element information
        elemInfo = {
            "type": elem_type,
            "nOfNodes": T.shape[1],
            "faceNodes": face_nodes.tolist() if face_nodes is not None else None,
            "faceNodes1d": face_nodes1d.tolist() if face_nodes1d is not None else None,
        }

        # Save as .npz file
        npz_file_name = os.path.splitext(dcm_file)[0] + ".npz"
        npz_file_path = os.path.join(directory, npz_file_name)

        np.savez(npz_file_path, X=X, T=T, elemInfo=elemInfo, **Tb)

        npz_files.append(npz_file_name)

    return npz_files


print("Function loaded:", generate_python_mesh_file)

# OBTAIN computational meshes (do not work) this would be written in main

"""import importlib
import mesh.mesh_converter
importlib.reload(mesh.mesh_converter)  # Force recharge of the module
from mesh.mesh_converter import generate_python_mesh_file

npz_files = generate_python_mesh_file("mesh")
print("files:",npz_files)

mesh_data = np.load("mesh/mesh1_P1.npz")
print(mesh_data.files)
X = mesh_data["X"]
T = mesh_data["T"]
Tb = mesh_data["Tb_Other"]
elem_info_data = np.load("mesh/mesh1_P1.npz", allow_pickle=True)
elemInfo = elem_info_data["elemInfo"].item()  # Convierte el objeto almacenado en un diccionario de Python

print("X:\n", X)
print("T:\n", T)
print("Tb:\n", Tb)
print("elemInfo:", elemInfo)"""
