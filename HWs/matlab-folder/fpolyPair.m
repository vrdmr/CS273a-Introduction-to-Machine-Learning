function Xext = fpoly(X, degree, useConstant)
% X=fpoly(X,degree,useConst) : create polynomial features of each individual feature (too many cross products)
%   useConstant=0 => do not include "constant" feature, x_0^(j) = 1 for all j

if (nargin < 3 ) useConstant = 1; end;
[m,n]=size(X);
%if (n~=1) error('PolyX currently only works for univariate data'); end;


npoly = ceil((n^(degree+1) - 1)/(n-1));   % ceil to fix possible roundoff error
if (useConstant)
  Xext = zeros(m,npoly);
  Xext(:,1) = 1;
  Xcur = 1;
  k=2;
else 
  Xext = zeros(m,npoly-1);
  Xcur = 1;
  k=1;
end;

% Hard coded to be a shorter length
if (degree==2)
  Xext(:,k:k+n-1) = X; k=k+n;
  X2 = bsxfun(@times,X,reshape(X,[m,1,n]));
  X2 = reshape(X2,[m,n*n]);
  idx = find( bsxfun(@ge,(1:n)',1:n) )';
  K=length(idx);
  Xext(:,k:k+K-1) = X2(:,idx);
  Xext = Xext(:,1:k+K-1);
  return;
end;


for p=1:degree,
  Xcur = bsxfun(@times,X,Xcur);
  Xcur = reshape(Xcur,m,numel(Xcur)/m);
  K = size(Xcur,2);
  k,K,size(Xext(:,k:k+K-1)), size(Xcur),
  Xext(:,k:k+K-1) = Xcur;
  k=k+K;
  Xcur = reshape(Xcur,[m,1,K]);
end;

