function [Z W] = fkitchensink(X, K, type, W)
% random kitchen sink features from data
% [X, W] = fkitchensink(X, K, type [,W]) selects K random "kitchen sink" features of X
%   "type" is one of: 'stump', 'sigmoid', 'sinusoid', 'linear'
%   optional return value "W" are the random parameters; optional argument "W" uses fixed params
%   Ex: [Xtr, W] = fkitchensink(Xtr,10,'sigmoid'); Xte = fkitchensink(Xte,[],'sigmoid',H);  
% see also: fpoly, fhash, fproject, fsvd
  [N,M] = size(X);
  if (nargin < 4)
    switch lower(type)
    case 'stump',  
      W=zeros(2,K);  
      s = sqrt(var(X));
      W(1,:)=ceil(rand(1,K)*M);         % random feature index 1..M
      W(2,:)=randn(1,K) .* s(W(1,:));   % random threshold (w/ same variance as that feature)
    case {'sigmoid', 'sinusoid', 'linear'}
      W=randn(M,K);                     % random direction for sigmoidal ridge
                                        % random freq for sinusoids; random linear projections
    end;
  end;
  if (isempty(K)) K=size(W,2); end;

  % evaluate the features:
  Z = zeros(N,K);
  switch lower(type)
  case 'stump',                           % decision stump with random threshold
    for i=1:K, 
      Z(:,i)= X(:,W(1,i)) >= W(2,i); 
    end;
  case 'sigmoid'                          % sigmoidal ridge with random direction
    Z = X*W;
    Z = 1./(1+exp(Z));
  case 'sinusoid'                         % sinusoid with random frequency
    Z = sin(X*W);
    %Z = [cos(X*W) sin(X*W)];
  case 'linear'                           % straight linear projection
    Z = X*W;
  end;

