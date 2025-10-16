function plotPostprocessedSolution(X,T,u,referenceElementStar,nDegRef)

% Check input
if nargin == 4
    nDegRef = 20;
end

% Plotting element (equal spaced points)
nodes = [];
h = 1/nDegRef;
for j = 0:nDegRef
    i = (0:nDegRef-j)';
    aux = j*ones(size(i));
    nodes = [nodes; [i aux]*h];
end
nodes = 2*nodes - 1;

% Delaunay triangulation of plotting element
elemTriRef = delaunayn(nodes);

coordRef = referenceElementStar.NodesCoord;
nDeg = referenceElementStar.degree;

nOfElementNodes = size(coordRef,1);
nOfElements = size(T,1);

% Compute (k+1)-degree shape functions at interpolation points
shapeFunctions=evaluateNodalBasisTriwithoutDerivatives(nodes,referenceElementStar.NodesCoord,nDeg);

% Compute k-degree shape functions at interpolation points
shapeFunctionsGeo=evaluateNodalBasisTriwithoutDerivatives(nodes,referenceElementStar.NodesCoordGeo,nDeg-1);

% Loop in elements
hold on
for ielem = 1:nOfElements
    % Interpolate solution and position at interpolation points
    Xplot = shapeFunctionsGeo*X(T(ielem,:),:);
    ind = (ielem-1)*nOfElementNodes+1:ielem*nOfElementNodes;
    uplot = shapeFunctions*u(ind);

    % Plot interpolated solution in the element    
    trisurf(elemTriRef,Xplot(:,1),Xplot(:,2),uplot)    
end
hold off
axis equal
shading interp
colormap jet

