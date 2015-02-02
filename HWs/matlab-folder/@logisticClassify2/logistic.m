function value = logistic(obj, X)
% logistic(obj,X): evaluate the logistic function for weights obj.wts, data X
% wts is (1 x d+1), X is (n x d)  : d features, n data points
%
  [n,d] = size(X);
  X = [ones(n,1), X];      % extend features X by the constant feature
  f = X * obj.wts' ;       % compute weighted sum of features
  value = 1./(1+exp(-f));  % evaluate logistic function of weighted sum

