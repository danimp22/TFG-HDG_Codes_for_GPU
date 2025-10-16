function [KK, f, QQ, UU, Qf, Uf] = hdgMatrixPoisson(muElem,X,T,F,referenceElement,infoFaces,tau)

nOfFaces = max(max(F));
nOfElements = size(T,1);
nOfInteriorFaces = size(infoFaces.intFaces,1);
nOfFaceNodes = size(referenceElement.NodesCoord1d,1);
nDOF = nOfFaces*nOfFaceNodes;
f = zeros(nDOF,1);
QQ = cell(nOfElements,1); UU = cell(nOfElements,1); Qf = cell(nOfElements,1); Uf = cell(nOfElements,1);

% loop in elements
indK = 1; n = nOfElements*(3*nOfFaceNodes)^2; ind_i  = zeros(1,n); ind_j  = zeros(1,n); coef_K = zeros(1,n);
for iElem = 1:nOfElements
    Te = T(iElem,:);
    Xe = X(Te,:);
    Fe = F(iElem,:);
    isFeInterior = (Fe <= nOfInteriorFaces); %Boolean (1=interior face)

    % elemental matrices
    [Qe,Ue,Qfe,Ufe,Alq,Alu,All] = KKeElementalMatricesIsoParametric(muElem(iElem),Xe,referenceElement,tau(iElem,:));
    
    % Interior faces seen from the second element are flipped to have
    % proper orientation
    flipFace = zeros(1,3); %Boolean (1=face to be flipped)
    flipFace(isFeInterior)=any(infoFaces.intFaces(Fe(isFeInterior),[1 3])<iElem,2);
    indL=1:3*nOfFaceNodes; 
    aux=nOfFaceNodes:-1:1; indflip=[aux,nOfFaceNodes+aux,2*nOfFaceNodes+aux];
    aux=ones(1,nOfFaceNodes); aux = [flipFace(1)*aux, flipFace(2)*aux, flipFace(3)*aux];
    indL(aux==1)=indflip(aux==1); %permutation for local numbering
    
    Qe=Qe(:,indL);    Ue=Ue(:,indL);
    Alq=Alq(indL,:);  Alu=Alu(indL,:);   All=All(indL,indL);
    
    %The local problem solver is stored for postprocess
    QQ{iElem} = sparse(Qe);  UU{iElem} = sparse(Ue);
    Qf{iElem} = sparse(Qfe); Uf{iElem} = sparse(Ufe);
    
    %Elemental matrices to be assembled
    KKe = Alq*Qe + Alu*Ue + All;
    ffe = -(Alq*Qfe + Alu*Ufe);


    
    aux = (1:nOfFaceNodes);
    indRC = [(Fe(1)-1)*nOfFaceNodes + aux,(Fe(2)-1)*nOfFaceNodes + aux,(Fe(3)-1)*nOfFaceNodes + aux];
    f(indRC) = f(indRC) + ffe;
    for irow = 1:(3*nOfFaceNodes)
        for icol = 1:(3*nOfFaceNodes)
            ind_i(indK)  = indRC(irow); ind_j(indK)  = indRC(icol);
            coef_K(indK) = KKe(irow,icol); indK = indK+1;
        end
    end
end
KK = sparse(ind_i,ind_j,coef_K);


%%
%% ELEMENTAL MATRIX
function [Q,U,Qf,Uf,Alq,Alu,All] = KKeElementalMatricesIsoParametric(mu,Xe,referenceElement,tau)

nOfElementNodes = size(referenceElement.NodesCoord,1);
nOfFaceNodes = size(referenceElement.NodesCoord1d,1);
faceNodes = referenceElement.faceNodes;
nOfFaces = 3; %triangles

% Information of the reference element
N = referenceElement.N;
Nxi = referenceElement.Nxi; Neta = referenceElement.Neta;
N1d = referenceElement.N1d; Nx1d = referenceElement.N1dxi;
%Numerical quadrature
IPw_f = referenceElement.IPweights1d; ngf = length(IPw_f);
IPw = referenceElement.IPweights; ngauss = length(IPw);

%%Volume computations
% Jacobian
J11 = Nxi*Xe(:,1); J12 = Nxi*Xe(:,2); 
J21 = Neta*Xe(:,1); J22 = Neta*Xe(:,2); 
detJ = J11.*J22-J12.*J21;
%maybe we should use bsxfun instead of diagonal matrices...
dvolu = spdiags(referenceElement.IPweights.*detJ,0,ngauss,ngauss);
invJ11 = spdiags(J22./detJ,0,ngauss,ngauss);
invJ12 = spdiags(-J12./detJ,0,ngauss,ngauss);
invJ21 = spdiags(-J21./detJ,0,ngauss,ngauss);
invJ22 = spdiags(J11./detJ,0,ngauss,ngauss);
% xy-derivatives
Nx = invJ11*Nxi + invJ12*Neta;
Ny = invJ21*Nxi + invJ22*Neta;

%Computation of r.h.s. source term (analytical laplacian)
Xg = N*Xe;
sourceTerm = sourcePoisson(Xg,mu);
fe = N'*(dvolu*sourceTerm);
%Elemental matrices
Me = N'*(dvolu*N);
Aqq = zeros(2*size(Me));
aux = 1:2:2*nOfElementNodes; aux2 = 2:2:2*nOfElementNodes;
Aqq(aux,aux)=Me; Aqq(aux2,aux2)=Me;
Auq = zeros(nOfElementNodes,2*nOfElementNodes);
Auq(:,aux) = N'*(dvolu*Nx); %x derivatives & 1st component of q
Auq(:,aux2)= N'*(dvolu*Ny); %y derivatives & 2nd component of q

%% Faces computations
Alq = zeros(3*nOfFaceNodes,2*nOfElementNodes);
Auu = zeros(nOfElementNodes,nOfElementNodes);
Alu = zeros(3*nOfFaceNodes,nOfElementNodes);
All = zeros(3*nOfFaceNodes,3*nOfFaceNodes);
%Is it possible to remove this loop?
for iface = 1:nOfFaces    
    tau_f = tau(iface);
    nodes = faceNodes(iface,:); Xf = Xe(nodes,:); % Nodes in the face
    dxdxi = Nx1d*Xf(:,1); dydxi = Nx1d*Xf(:,2);
    dxdxiNorm = sqrt(dxdxi.^2+dydxi.^2);
    dline = dxdxiNorm.*IPw_f';
    nx = dydxi./dxdxiNorm; ny=-dxdxi./dxdxiNorm;
    %Face matrices
    ind_face = (iface-1)*nOfFaceNodes + (1:nOfFaceNodes);
    Alq(ind_face,2*nodes-1) = N1d'*(spdiags(dline.*nx,0,ngf,ngf)*N1d);
    Alq(ind_face,2*nodes) = N1d'*(spdiags(dline.*ny,0,ngf,ngf)*N1d);
    Auu_f = N1d'*(spdiags(dline,0,ngf,ngf)*N1d)*tau_f;  
    Auu(nodes,nodes) = Auu(nodes,nodes) + Auu_f;
    Alu(ind_face,nodes) = Alu(ind_face,nodes) + Auu_f;
    All(ind_face,ind_face) = -Auu_f;
end

% Elemental mapping
Aqu = -mu*Auq'; Aul = -Alu'; Aql = mu*Alq';
A = [Auu Auq; Aqu Aqq];
UQ = -A\[Aul;Aql];
fUQ= A\[fe;zeros(2*nOfElementNodes,1)];

U = UQ(1:nOfElementNodes,:); 
Uf=fUQ(1:nOfElementNodes); % maps lamba into U

Q = UQ(nOfElementNodes+1:end,:); 
Qf=fUQ(nOfElementNodes+1:end); % maps lamba into Q
