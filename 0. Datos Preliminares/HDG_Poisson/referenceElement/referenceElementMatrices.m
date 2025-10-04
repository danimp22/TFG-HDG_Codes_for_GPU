function [M,C,m] = referenceElementMatrices(referenceElement, nOfElemNodes, nOfFaceNodes)

M = zeros(nOfElemNodes);
C = zeros(nOfElemNodes,nOfElemNodes,2);
for iNode = 1:nOfElemNodes
    for jNode = 1:nOfElemNodes
        M(iNode,jNode) = referenceElement.N(:,iNode)'*(referenceElement.N(:,jNode).*referenceElement.IPweights);
        C(iNode,jNode,1) = referenceElement.N(:,iNode)'*(referenceElement.Nxi(:,jNode).*referenceElement.IPweights);
        C(iNode,jNode,2) = referenceElement.N(:,iNode)'*(referenceElement.Neta(:,jNode).*referenceElement.IPweights);
    end
end

m = zeros(nOfFaceNodes);
for iNode = 1:nOfFaceNodes
    for jNode = 1:nOfFaceNodes
        m(iNode,jNode) = referenceElement.N1d(:,iNode)'*(referenceElement.N1d(:,jNode).*referenceElement.IPweights1d');
    end
end
