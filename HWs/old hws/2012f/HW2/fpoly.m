function Xext = fpoly(X, degree, useConstant)
% X=fpoly(X,degree,useConst) : create polynomial features of each individual feature (no cross products)
%   useConstant=0 => do not include "constant" feature, x_0^(j) = 1 for all j

if (nargin < 3 ) useConstant = 1; end;
[m,n]=size(X);
%if (n~=1) error('PolyX currently only works for univariate data'); end;


if (useConstant)
  Xext = zeros(m,n*degree+1);
  Xext(:,1) = 1;
  k=2;
else 
  Xext = zeros(m,n*degree);
  k=1;
end;

for p=1:degree,
  for j=1:n,
    Xext(:,k)=X(:,j).^p;
    k=k+1;
  end;
end;

