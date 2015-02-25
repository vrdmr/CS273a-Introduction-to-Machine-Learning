function [X mu sig] = whiten(X, mu,sig)
% [X, mu,sig] = whiten(X [, mu,sig]) whitens X to be zero mean, uncorrelated and unit variance
%   optional return values are the transform; optional arguments use fixed transform values instead
%   Ex: [Xtr, m,s]=whiten(Xtr); Xte=whiten(Xte,m,s);  % whitens training data & changes test data to match
% see also: rescale
  if (nargin < 2)
    mu = mean(X);
    C = cov(X);
    [U,S,V] = svd(C);
    sig = U*diag(1./sqrt(diag(S)));
  end;
  X = bsxfun(@minus, X, mu);
  X = X*sig;

