function u_star = HDGpostprocess(X,T,u,q,referenceElementStar)
% q is assumed to be grad(u) in two columns  

nOfElements = size(T,1);
nOfElementNodes = size(T,2);
coordRef_star = referenceElementStar.NodesCoord;
npoints = size(coordRef_star,1);

% Compute shape functions at interpolation points
nDeg = referenceElementStar.degree-1;
shapeFunctions=evaluateNodalBasisTriwithoutDerivatives(referenceElementStar.NodesCoord,referenceElementStar.NodesCoordGeo,nDeg);

% u star initialization
u_star = zeros(npoints*nOfElements,1);

% Loop in elements
for iElem = 1:nOfElements
    
    ind = (iElem-1)*nOfElementNodes+1:iElem*nOfElementNodes;
    ind_star = (iElem-1)*npoints+1:iElem*npoints;
    ue = shapeFunctions*u(ind);
    qe = shapeFunctions*q(ind,:);
    [Ke,Beqe,int_ue_star, int_ue] = ElementalMatrices(X(T(iElem,:),:),referenceElementStar,ue,qe);
    
    % Lagrange multipliers
    K = [Ke int_ue_star; int_ue_star' 0];
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
Nxi = referenceElement_star.Nxi;
NxiGeo=referenceElement_star.dNGeodxi;
Neta = referenceElement_star.Neta;
NetaGeo = referenceElement_star.dNGeodeta;
NN_xy = zeros(1,2*nOfElementNodes_star);

% Number of Gauss points in the interior
ngauss = length(IPw);

% x and y coordinates of the element nodes
xe = Xe(:,1); ye = Xe(:,2);

% VOLUME COMPUTATIONS: LOOP IN GAUSS POINTS
for g = 1:ngauss
    %Shape functions and derivatives at the current integration point
    N_g = N(g,:);
    Nxi_g = Nxi(g,:);
    Neta_g = Neta(g,:);
    
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
    qx_g = N_g*qe(:,1);
    qy_g = N_g*qe(:,2);
    
    %Contribution of the current integration point to the elemental matrix
    Bq = Bq + (Nx_g'*qx_g + Ny_g'*qy_g)*dvolu; 
    K = K + (Nx_g'*Nx_g + Ny_g'*Ny_g)*dvolu;
    int_u_star = int_u_star + N_g'*dvolu;
    int_u = int_u + u_g*dvolu;
end


