function [X feat] = fsubset(X, K, feat)
% select subset of features from data
% [X, feat] = fsubset(X, K [,feat]) selects a fixed or random subset of K features from X
%   optional return value "feat" is the features; optional argument "feat" uses fixed features
%   Ex: [Xtr, F] = fsubset(Xtr,10); Xte = fsubset(Xte,[],F);  % chooses random features from Xtr & Xte
% see also: 
  [N,M] = size(X);
  if (nargin < 3)
    feat = randperm(M); feat=feat(1:K);
  end;
  X = X(:,feat);

