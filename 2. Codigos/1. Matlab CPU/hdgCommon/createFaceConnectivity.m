function F = createFaceConnectivity(T)
nOfNodes = max(max(T));
nOfElements = size(T,1);
CheckFaces = zeros(nOfNodes);
F = zeros(nOfElements,3);
facesIndex = 0;
aux = 1:3;
for iElem = 1:size(T,1)
    Te = T(iElem,1:3);
    f1 = Te([1 2]);
    f2 = Te([2 3]);
    f3 = Te([3 1]);
    bolf1 = CheckFaces(f1(1),f1(2))~=0 | CheckFaces(f1(1),f1(2))~=0;
    bolf2 = CheckFaces(f2(1),f2(2))~=0 | CheckFaces(f2(1),f2(2))~=0;
    bolf3 = CheckFaces(f3(1),f3(2))~=0 | CheckFaces(f3(1),f3(2))~=0;
    bol = [bolf1 bolf2 bolf3];
    facesIndex = facesIndex(end)+(1:length(aux(~bol)));
    F(iElem,~bol) = facesIndex;
    prevEl_1 = CheckFaces(f1(1),f1(2));
    prevEl_2 = CheckFaces(f2(1),f2(2));
    prevEl_3 = CheckFaces(f3(1),f3(2));
    F(iElem,bol) = [prevEl_1(prevEl_1~=0) prevEl_2(prevEl_2~=0) prevEl_3(prevEl_3~=0)];
    CheckFaces(f1,f1) =  F(iElem,1);
    CheckFaces(f2,f2) =  F(iElem,2);
    CheckFaces(f3,f3) =  F(iElem,3);
end