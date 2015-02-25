function [X Y] = dataGauss(N0,N1,mu0,mu1,Sig0,Sig1)
if (nargin < 2) N1=N0; end;
if (nargin < 3) mu0 = [0 0]; end;
if (nargin < 4) mu1 = [1 1]; end;
if (nargin < 5) Sig0 = eye(2); end;
if (nargin < 6) Sig1 = eye(2); end;

d = size(mu0,2);
if (size(mu1,2) ~= d || any(size(Sig0)~=[d d]) || any(size(Sig1)~=[d d]))
  error('Dimensions should agree'); 
end;

X0 = randn(N0,d)*sqrtm(Sig0)+ones(N0,1)*mu0; Y0 = -ones(N0,1);
X1 = randn(N1,d)*sqrtm(Sig1)+ones(N1,1)*mu1; Y1 = ones(N1,1);

X=[X0;X1]; Y=[Y0;Y1]; 


