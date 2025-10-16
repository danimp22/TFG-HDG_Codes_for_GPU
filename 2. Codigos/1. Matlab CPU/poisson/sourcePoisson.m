function s = sourcePoisson(X, mu)

% Parameters
lambda = 10;
f = 6;

% Points
x = X(:,1);
y = X(:,2);

% Minus Laplacian

s=-mu*(2+0.*x);
%s = mu*(4*lambda^4*y.*exp(-lambda*y).*cos(pi*f*x) - 8*lambda^3*exp(-lambda*y).*cos(pi*f*x) - ...
%      4*lambda^3*exp(-2*lambda*y) - 4*pi^2*f^2*lambda^2*y.*exp(-lambda*y).*cos(pi*f*x) - 8);