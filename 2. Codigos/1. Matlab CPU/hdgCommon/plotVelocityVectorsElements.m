function plotVelocityVectorsElements(X,T,u)

Xs=[];
for i=1:size(T,1), Xs=[Xs;X(T(i,:),:)]; end
quiver(Xs(:,1),Xs(:,2),u(:,1),u(:,2));