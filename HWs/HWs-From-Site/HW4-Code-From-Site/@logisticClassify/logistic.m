function s = logistic(obj, X)
% logistic(obj,X): evaluate the (multi-)logistic function for weights obj.wts, data X
% wts is (C-1 x d+1), X is (n x d)  : d features, n data points, c classes
%
  [n,d] = size(X);
  X = [ones(n,1), X];      % extend features X by the constant feature
  z = X * obj.wts' ;       % compute weighted sum of features
  z = [zeros(n,1), z];     % add "class 0 response"
  z = bsxfun(@minus, z, max(z,[],2));  % subtract max to make exp more numerically stable
  s = exp(z);
  s = bsxfun(@rdivide, s, sum(s,2)); % evaluate multi-logit function of the responses

