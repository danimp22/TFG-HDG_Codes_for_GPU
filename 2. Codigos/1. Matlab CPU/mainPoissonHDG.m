% Academic 2D HDG code for solving the Poisson equation with Dirichlet 
% boundary conditions.
%
% First version by Sonia Fernandez-Mendez LaCaN UPC-BarcelonaTech 2016
% Second version by Ruben Sevilla, Swansea University 2017
%
% www.lacan.upc.edu
%
% Main data variables:
%  X: nodal coordinates
%  T: mesh connectivitity matrix
%  F: faces (here sides) for each element 
%    (Faces are numbered so that interior faces are first)
%  elemInfo: element type
%  infoFaces.intFaces: [elem1 face1 elem2 face2 rotation] for each face
%    (Each face is assumed to have the orientation given by the 
%    first element, it is flipped when seen from the second element)
%  infoFaces.extFaces: [elem1 face1] for each exterior face
%  referenceElement: integration points and shape functions (volume and
%    boundary sides)
%

clearvars
close all
setpath
global mu
mu=1;
  
%% Load computational mesh 
degree=1; 
load(['mesh1_P',num2str(degree)]); 
 
nOfElements = size(T,1);

% figure(1),clf,plotMesh(X,T);

%% HDG preprocess
[F, infoFaces] = hdg_preprocess(T);
nOfFaces = max(max(F)); 

%% Viscosity parameter
muElem = mu*ones(nOfElements,1);

%% Stabilization parameter
tau = ones(nOfElements,3); %Uniform tau parameter (all faces)
%tau(:,2:3) = 0; %Non-null only for 1st face in each element

%% Computation
% Loop in elements
referenceElement=createReferenceElementTri(degree);
disp('Loop in elements...')
[K,f, QQ, UU, Qf, Uf] = hdgMatrixPoisson(muElem,X,T,F,referenceElement,infoFaces,tau);

%Dirichlet BC
%Dirichlet face nodal coordinates
nOfFaceNodes = degree+1; 
nOfInteriorFaces = size(infoFaces.intFaces,1);
nOfExteriorFaces = size(infoFaces.extFaces,1); 

uDirichlet = computeProjectionFaces(@analyticalPoisson,infoFaces.extFaces,X,T,referenceElement);
dofDirichlet= nOfInteriorFaces*nOfFaceNodes + (1:nOfExteriorFaces*nOfFaceNodes);
dofUnknown = 1:nOfInteriorFaces*nOfFaceNodes;

% System reduction (Dirichlet faces  are set to prescribed value)
f = f(dofUnknown)-K(dofUnknown,dofDirichlet)*uDirichlet;
K = K(dofUnknown,dofUnknown);

% Face solution
disp('Solving linear system...')
lambda = K\f; 
uhat = [lambda(1:nOfInteriorFaces*nOfFaceNodes); uDirichlet];

% Elemental solution
disp('Calculating element by element solution...')
[u,q]=computeElementsSolution(uhat,UU,QQ,Uf,Qf,F);

figure(2),clf
plotDiscontinuosSolution(X,T,u,referenceElement,20)
colorbar, title('HDG solution: u')

% Local postprocess for superconvergence 
 disp('Performing local postprocess...')
 referenceElement_star = createReferenceElementTriStar(referenceElement);
 u_star = HDGpostprocess(X,T,u,-q,referenceElement_star);

%Plots postprocess solution
figure(22),clf
plotPostprocessedSolution(X,T,u_star,referenceElement_star,20);
colorbar, title('HDG solution: u*')

%% Errors

% analytical solution
u_ex = @analyticalPoisson;

%Error for the HDG solution
Error=computeL2Norm(referenceElement,X,T,u,u_ex);
fprintf('Error HDG = %e\n',Error);

% Error for the postprocessed solution
ErrorPost=computeL2NormPostprocess(referenceElement_star,X,T,u_star,u_ex);
fprintf('Error HDG postprocessed = %e\n',ErrorPost);
disp(' ')


