function u_star = HDGpostprocessSuperParametric(muElem,X,T,u,q,referenceElement_star,referenceElement)

nOfElements = size(T,1);
coordRef=referenceElement.NodesCoord;
nOfElementNodes = size(coordRef,1);
coordRef_star = referenceElement_star.NodesCoord;
npoints = size(coordRef_star,1);
degree=referenceElement.degree;

%Shape functions for interpolation of nodal values
[N,dNdxi,dNdeta]=evaluateNodalBasisTri(coordRef_star,coordRef,degree);

% u star initialization
u_star = zeros(npoints*nOfElements,1);

% Loop in elements
for iElem = 1:nOfElements   
    ind = (iElem-1)*nOfElementNodes+1:iElem*nOfElementNodes;
    ind_star = (iElem-1)*npoints+1:iElem*npoints;
    ue = N*u(ind);
    qe(1:2:2*npoints) = N*q(2*ind-1);
    qe(2:2:2*npoints) = N*q(2*ind);
    [Ke,Beqe,int_ue_star, int_ue] = ElementalMatrices(X(T(iElem,:),:),referenceElement_star,ue,qe);
    
    % Lagrange multipliers
    K = [muElem(iElem)*Ke int_ue_star; int_ue_star' 0];
    f = [Beqe;int_ue];
    
    % elemental solution
    sol = K\f;
    
    % postprocessed solution
    u_star(ind_star) = sol(1:end-1);
end

%%
%% ELEMENTAL MATRIX

function [K,Bq,int_u_star,int_u] = ElementalMatrices(Xe,referenceElement_star,ue,qe)

nOfElementNodes_star = size(referenceElement_star.NodesCoord,1);
K = zeros(nOfElementNodes_star,nOfElementNodes_star);
Bq = zeros(nOfElementNodes_star,1);
int_u_star = zeros(nOfElementNodes_star,1);
int_u = 0;

% Information of the reference element
IPw = referenceElement_star.IPweights;
N = referenceElement_star.N;
Nxi = referenceElement_star.Nxi; Neta = referenceElement_star.Neta;
NxiGeo = referenceElement_star.NxiGeo; NetaGeo = referenceElement_star.NetaGeo;
NN_xy = zeros(1,2*nOfElementNodes_star);

% Number of Gauss points in the interior
ngauss = length(IPw);

% x and y coordinates of the element nodes
xe = Xe(:,1); ye = Xe(:,2);

% VOLUME COMPUTATIONS: LOOP IN GAUSS POINTS
for g = 1:ngauss
    %Shape functions and derivatives at the current integration point
    N_g = N(g,:);    Nxi_g = Nxi(g,:);    Neta_g = Neta(g,:);
    
    %Jacobian
    J = [NxiGeo(g,:)*xe	  NxiGeo(g,:)*ye
        NetaGeo(g,:)*xe  NetaGeo(g,:)*ye];
    
    %Integration weight
    dvolu=IPw(g)*det(J);
    
    %x and y derivatives
    invJ = inv(J); 
    Nx_g = invJ(1,1)*Nxi_g + invJ(1,2)*Neta_g;
    Ny_g = invJ(2,1)*Nxi_g + invJ(2,2)*Neta_g;
    NN_xy(1:2:end) = Nx_g;
    NN_xy(2:2:end) = Ny_g;
    
    % u and q at gauss points
    u_g = N_g*ue;
    qx_g = N_g*qe(1:2:end)';
    qy_g = N_g*qe(2:2:end)';
    
    %Contribution of the current integration point to the elemental matrix
    Bq = Bq - (Nx_g'*qx_g + Ny_g'*qy_g)*dvolu; 
    K = K + (Nx_g'*Nx_g + Ny_g'*Ny_g)*dvolu;
    int_u_star = int_u_star + N_g'*dvolu;
    int_u = int_u + u_g*dvolu;
end


