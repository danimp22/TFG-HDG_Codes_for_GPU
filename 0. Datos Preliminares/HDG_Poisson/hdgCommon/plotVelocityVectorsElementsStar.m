function plotVelocityVectorsElementsStar(X,T,u,referenceElement_star)

%Basis functions of standard element at star element nodes for
%interpolation of X
N=evaluateNodalBasisTriwithoutDerivatives(referenceElement_star.NodesCoord,referenceElement_star.NodesCoordGeo,referenceElement_star.degree-1);

Xs=[];
for i=1:size(T,1), Xs=[Xs;N*X(T(i,:),:)]; end
quiver(Xs(:,1),Xs(:,2),u(:,1),u(:,2));