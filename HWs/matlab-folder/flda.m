function [Xlda, T] = flda(X, Y, K, T)
% [X,T]=flda(X,Y,K [,T]) : reduce the dimension of X to K features using (multiclass) linear discriminant analysis
% optional return value T represents the transform; optional argument T uses T instead of computing the LDA

[m,n]=size(X);

if (nargin < 4)
  c = unique(Y);
  nc = zeros(1,length(c));
  mu = zeros(length(c),n);
  sig= zeros(length(c),n,n);
  for i=1:length(c),
    idx = find(Y==c(i)); 
    nc(i) = length(idx);
    mu(i,:) = mean(X(:,idx));
    sig(i,:,:) = cov(X(:,idx));
  end;
  S = (nc/n) * reshape(sig, length(c),n*n); 
  S = reshape(S, n,n);
  M = cov(mu);
  
  [U,S,V] = svds(X,K);              % compute svd 
  Xsvd = U*sqrt(S);                 % new data coefficients
  T = sqrt(S(1:K,1:K))*V';          % new bases for data
else
  Xsvd = X/T;                       % or, use given set of bases
end;


