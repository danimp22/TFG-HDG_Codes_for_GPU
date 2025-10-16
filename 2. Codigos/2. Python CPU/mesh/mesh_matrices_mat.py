import scipy.io
import numpy as np

def load_mesh(matrix_number, degree, directory="mesh/"):
    """
    Loads a MATLAB .mat file based on the user-specified matrix number and degree
    and initializes the matrices as Python variables with 0-indexed connectivity.
    
    Parameters:
        matrix_number (int): The matrix index (e.g., 1, 2, 3, ...).
        degree (int): The polynomial degree used to construct the filename.
        directory (str): Directory where the .mat files are stored.

    Returns:
        tuple: (X, T, Tb_Dirichlet) as NumPy arrays, with T and Tb_Dirichlet adjusted to 0-based indexing.
    """
    # Construct the file path dynamically.
    file_path = f"{directory}mesh{matrix_number}_P{degree}.mat"

    try:
        # Load the .mat file.
        mat_data = scipy.io.loadmat(file_path)

        X = mat_data.get("X", None)
        T = mat_data.get("T", None)
        # The MATLAB file uses a misspelling: "Tb_Diriclet" instead of "Tb_Dirichlet".
        Tb_Dirichlet = mat_data.get("Tb_Diriclet", None)

        X = X if X is not None else np.array([])
        T = T if T is not None else np.array([])
        Tb_Dirichlet = Tb_Dirichlet if Tb_Dirichlet is not None else np.array([])

        # 1-indexed to 0-indexed.
        if T.size > 0:
            T = T.astype(int) - 1
        if Tb_Dirichlet.size > 0:
            Tb_Dirichlet = Tb_Dirichlet.astype(int) - 1

        print(f"Matrices loaded from '{file_path}' with 0-indexed connectivity.")
        return X, T, Tb_Dirichlet

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None, None




