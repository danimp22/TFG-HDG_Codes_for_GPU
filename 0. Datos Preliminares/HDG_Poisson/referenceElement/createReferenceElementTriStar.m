function referenceElementStar=createReferenceElementTriStar(referenceElement)

%reference element with degree k+1
referenceElementStar=createReferenceElementTri(referenceElement.degree+1);
%basis function of degree k at integration points
[NGeo,dNGeodxi,dNGeodeta]=evaluateNodalBasisTri(referenceElementStar.IPcoordinates,referenceElement.NodesCoord,referenceElement.degree);
referenceElementStar.NGeo=NGeo;
referenceElementStar.dNGeodxi=dNGeodxi;
referenceElementStar.dNGeodeta=dNGeodeta;
referenceElementStar.NodesCoordGeo=referenceElement.NodesCoord;