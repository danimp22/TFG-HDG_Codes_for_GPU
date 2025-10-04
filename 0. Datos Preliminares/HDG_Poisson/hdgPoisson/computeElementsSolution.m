function [u,q]=computeElementsSolution(lambda,UU,QQ,Uf,Qf,F)

nOfElements = size(F,1);
[nOfElementNodes,aux] = size(UU{1}); 
nOfFaceNodes=aux/3;

u = zeros(nOfElements*nOfElementNodes,1);
q = zeros(2*nOfElements*nOfElementNodes,1);

%Loop in elements
for ielem = 1:nOfElements
    Fe = F(ielem,:);
    aux = 1:nOfFaceNodes;
    ind = [(Fe(1)-1)*nOfFaceNodes + aux,(Fe(2)-1)*nOfFaceNodes + aux,(Fe(3)-1)*nOfFaceNodes + aux];
    
    u((ielem-1)*nOfElementNodes+(1:nOfElementNodes)) = UU{ielem}*lambda(ind) + Uf{ielem};
    q((ielem-1)*2*nOfElementNodes+(1:2*nOfElementNodes)) = QQ{ielem}*lambda(ind) + Qf{ielem};
end

 q = reshape(q,2,nOfElements*nOfElementNodes)';