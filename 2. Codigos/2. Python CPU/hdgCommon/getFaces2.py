import numpy as np

def get_faces(T):
    
    """
 
    [intFaces,extFaces] = GetFaces(T)
    (only for triangles and tetrahedrons)
    
    For every face i:
    intFaces(i,:)=[element1 nface1 element2 nface2 node1] for interior faces
    extFaces(i,:)=[element1 nface1] for exterior faces
    
    element1, element2:   number of the elements
    nface1, nface2:       number of face in each element
    node1:  number of node in the 2nd element that matches with the 1st node
            in the 1st element (with the numbering of the face)
    
    Input:
    T: connectivity of the mesh
    
    Output:
    intFaces,extFaces: interior and exterior faces
 
    """
    nElem, nen = T.shape  # Number of elements and nodes per element
    assert nen == 3, "Only triangular elements are supported."

    # Define the faces in each element
    Efaces = np.array([[0, 1], [1, 2], [2, 0]])  

    # Store visited faces
    face_map = {}
    intFaces = []
    extFaces = []

    # Iterate over all elements
    for iElem in range(nElem):
        for iFace in range(3):  # Each triangle has 3 faces
            nodesf = tuple(sorted(T[iElem, Efaces[iFace]]))  # Ensure unique face order (3,1)-->(1,3)

            if nodesf in face_map:  # Internal face found
                jElem, jFace = face_map[nodesf]
                intFaces.append([jElem, jFace, iElem, iFace, nodesf[0]])
                del face_map[nodesf]  # Remove from map (already processed)
            else:
                face_map[nodesf] = (iElem, iFace)  # Store as potential external face

    # Remaining faces in the map are external
    for nodesf, (iElem, iFace) in face_map.items():
        extFaces.append([iElem, iFace])

    return intFaces, extFaces

"""# Check
T = np.array([[1, 2, 3], [2, 4, 3]])  # Connectivity (1-based indexing)
intFaces, extFaces = get_faces(T)

print("Internal Faces:", intFaces)
print("External Faces:", extFaces)"""