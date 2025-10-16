function h = computeMinElementSize(X, T)
%
% h = computeMinElementSize(X, T)
%
% Approximation by minimum edge size
%

h = 1e10;

v1 = [1,2,3];
v2 = [2,3,1];
for iElem = 1:size(T,1)
    Xe = X(T(iElem,:),:);
    ds = sqrt( (Xe(v1,1)-Xe(v2,1)).^2 + (Xe(v1,2)-Xe(v2,2)).^2 );
    h = min(h, min(ds));
end