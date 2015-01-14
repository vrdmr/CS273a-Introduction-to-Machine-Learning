function [Z hash] = fhash(X, K, hash)
% random hash of features from data
% [X, hash] = fhash(X, K [,hash]) selects a fixed or random hash of K features from X
%   optional return value "hash" is the hash fn; optional argument "hash" uses fixed hash
%   Ex: [Xtr, H] = fhash(Xtr,10); Xte = fhash(Xte,[],H);  % chooses random hash for Xtr & Xte
% see also: 
  [N,M] = size(X);
  if (nargin < 3)
    hash = ceil(rand(1,M)*K); 
  end;
  % do the hashing:
  Z = zeros(N,K);
  for i=1:M, 
    Z(:,hash(i))=Z(:,hash(i))+X(:,i); 
  end;

