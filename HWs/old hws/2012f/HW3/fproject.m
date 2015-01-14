function [X proj] = fproject(X, K, proj)
% random projection of features from data
% [X, proj] = fproject(X, K [,proj]) selects a fixed or random linear projection of K features from X
%   optional return value "proj" is the projection; optional argument "proj" uses fixed projection
%   Ex: [Xtr, F] = fproject(Xtr,10); Xte = fproject(Xte,[],F);  % chooses random projection for Xtr & Xte
% see also: 
  [N,M] = size(X);
  if (nargin < 3)
    proj = randn(M,K); 
  end;
  X = X*proj;

