function [X scale] = rescale(X, scale)
% rescale data to unit variance
% [X, scale] = rescale(X [, scale]) scales X to be unit variance in each dimension
%   optional return value "scale" is the scaling factors; optional argument "scale" scales by "scale" instead
%   Ex: [Xtr, Scale] = rescale(Xtr); Xte = rescale(Xte,Scale);  % rescales by training data's magnitude
% see also: whiten
  if (nargin < 2)
    scale = 1./sqrt( var( X ) );
  end;
  scale( isinf(scale) )=1;     % avoid scaling constant (zero variance) features
  X = bsxfun(@times,X,scale);

