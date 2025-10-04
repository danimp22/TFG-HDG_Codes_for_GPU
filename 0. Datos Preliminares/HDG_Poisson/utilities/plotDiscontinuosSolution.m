function plotDiscontinuosSolution(X,T,u,referenceElement,nDegRef)


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

coordRef = referenceElement.NodesCoord;
nDeg = referenceElement.degree;
nOfElementNodes = size(coordRef,1);
nOfElements = size(T,1);

% Compute shape functions at interpolation points
N =evaluateNodalBasisTri(nodes,coordRef,nDeg);

% Loop in elements
hold on
for ielem = 1:nOfElements
    % Interpolate solution and position at interpolation points
    Te = T(ielem,:);
    Xplot = N*X(Te,:);
    ind = (ielem-1)*nOfElementNodes+1:ielem*nOfElementNodes;
    uplot = N*u(ind);

    % Plot interpolated solution in the element
    hold on
    trisurf(elemTriRef,Xplot(:,1),Xplot(:,2),uplot) 
end
hold off
axis equal
shading interp
colormap jet
