function [Xsvd, T] = fsvd(X, K, T)
% [X,T]=fsvd(X,K [,T]) : reduce the dimensionality of X to K features using singular value decomposition
% optional return value T represents the transform; optional argument T uses T instead of computing the SVD

[m,n]=size(X);

if (nargin < 3)
  [U,S,V] = svds(X,K);              % compute svd 
  Xsvd = U*sqrt(S);                 % new data coefficients
  T = sqrt(S(1:K,1:K))*V';          % new bases for data
else
  Xsvd = X/T;                       % or, use given set of bases
end;


