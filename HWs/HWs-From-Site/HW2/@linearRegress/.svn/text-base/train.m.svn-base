function obj=train(obj, Xtr,Ytr, reg)
% learner = train(learner, Xtrain, Ytrain, alpha) : train a linear regression predictor on the given data
%    alpha = L2 regularization penalty (default zero), e.g.,  1/m ||y - w x'||^2 + alpha * ||w||^2

  Xtr = [ones(size(Xtr,1),1) Xtr];    % extend features by including a constant feature
  if (nargin < 4)
    obj.theta = (Xtr\Ytr)';
  else
    [M,N]=size(Xtr);
    obj.theta = Ytr'*(Xtr/M)*inv(Xtr'*Xtr/M + reg*eye(N));
  end;

