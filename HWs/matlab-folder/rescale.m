function [X mu scale] = rescale(X, mu, scale)
% center and rescale data to unit variance
% [X, mu,scale] = rescale(X [, mu,scale]) shifts and scales X to be zero mean, unit variance in each dimension
%   optional return values "mu,scale" are the shift & scale factors; optional arguments specify these instead of data
%   Ex: [Xtr, M,S] = rescale(Xtr); Xte = rescale(Xte,M,S);  % rescales by training data's magnitude
% see also: whiten

  if (nargin < 2) 
    mu = mean( X );
  end
  if (nargin < 3)
    scale = 1./sqrt( var( X ) );
  end;
  scale( isinf(scale) )=1;     % avoid scaling constant (zero variance) features
  X = bsxfun(@minus,X,mu);
  X = bsxfun(@times,X,scale);

