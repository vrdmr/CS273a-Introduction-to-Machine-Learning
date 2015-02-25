function wts = initWeights(obj,X,Y,method)
%  wts = initWeights(obj,X,Y,method) : initialize weights in one of a number of "quick and dirty" ways
%    method = 'zeros'  : all zeros (default)
%             'random' : random, Gaussian coefficients
%             'regress': simple one-versus-all linear regression on a subsample
%             'bayes'  : simple Gaussian Bayes model

if (nargin < 4 || isempty(method)) method = 'zeros'; end;

[n,d] = size(X);
C = length(unique(Y));

switch(lower(method))
% Initialize to all zeros
case 'zeros',
  wts = zeros(C-1,d+1);

% Initialize randomly (Gaussian coefficents)
case 'random',
  wts = randn(C-1,d+1);

% Initialize with a "small" linear regression 
case 'regress',                                
  wts = zeros(C,d+1);
  idx=randperm(n); 
  idx=idx(1:min(max(4*d,100),n));            
  X1 = [ones(length(idx),1) X(idx,:)];
  invCov = inv( X1'*X1+.1*eye(d+1) );
  for c=1:C,
    wts(c,:) = ((2*(Y(idx)==c)-1)'*X1) * invCov;  % Class-c versus all regression
  end;
  wts = bsxfun(@minus, wts, wts(1,:));        % make 1st class' parameters zero
  wts = wts(2:end,:)/2;

% Initialize using a Gaussian Bayes with equal covariances
case 'bayes',                                 
  wts = zeros(C,d+1);
  mu  = zeros(C,d);
  Sig = .1*eye(d);                        
  for c=1:C,                                  % compute means of each class, and average covariance
    mu(c,:) = mean(X(Y==c,:),1);
    Sig = Sig + mean(Y==c)*cov(X(Y==c,:));
  end;
  Sig = inv(Sig);
  wts(:,2:end) = mu*Sig;                      % coefficients on features (x)
  wts(:,1) = -.5*sum(mu .* wts(:,2:end),2);   % constant / bias term
  wts = bsxfun(@minus, wts, wts(1,:));        % make 1st class' parameters zero
  wts = wts(2:end,:);

end;


