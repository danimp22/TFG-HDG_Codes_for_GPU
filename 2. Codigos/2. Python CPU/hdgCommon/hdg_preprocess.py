import numpy as np
import hdgCommon.getFaces
from hdgCommon.getFaces import get_faces

def hdg_preprocess(T):
    """
    Processes the mesh connectivity matrix T to create face connectivity F 
    and identify internal and external faces, using 0-indexed values.
    
    Parameters:
        T (numpy.ndarray): Mesh connectivity matrix (Mx3 for triangular elements), 0-indexed.
    
    Returns:
        F (numpy.ndarray): Face connectivity matrix (Mx3) with 0-indexed face IDs.
        infoFaces (dict): Dictionary containing 'intFaces' and 'extFaces',
                          with entries in 0-indexed format:
                          - intFaces rows: [element1, face1, element2, face2, common_node]
                          - extFaces rows: [element, face]
    """
    # Get internal and external faces using the first 3 columns of T.
    # (Assume get_faces now returns 0-indexed information.)
    intFaces, extFaces = get_faces(T[:, :3])
    
    # Number of elements.
    nOfElements = T.shape[0]
    
    # Initialize face connectivity matrix F (each element has 3 faces).
    F = np.zeros((nOfElements, 3), dtype=int)
    
    # Assign internal faces: enumerate from 0.
    # intFaces rows are of the form: [element1, face1, element2, face2, common_node] (all 0-indexed)
    for iFace, infoFace in enumerate(intFaces, start=0):
        F[ infoFace[0], infoFace[1] ] = iFace  # Assign face ID to first element.
        F[ infoFace[2], infoFace[3] ] = iFace  # Assign same face ID to second element.
    
    # Assign external faces: enumerate from 0, but offset by the number of internal faces.
    for iFace, infoFace in enumerate(extFaces, start=0):
        F[ infoFace[0], infoFace[1] ] = iFace + len(intFaces)
    
    # Store the face information in a dictionary.
    infoFaces_dict = {
        "intFaces": intFaces,
        "extFaces": extFaces
    }
    
    return F, infoFaces_dict

"""# Example usage:
if __name__ == "__main__":
    # Example connectivity matrix for a simple mesh with two triangles (0-indexed)
    T = np.array([[0, 1, 2],
                  [1, 3, 2]])
    F, infoFaces = hdg_preprocess(T)
    print("Face connectivity matrix (F):")
    print(F)
    print("Face information (infoFaces):")
    print(infoFaces)"""



