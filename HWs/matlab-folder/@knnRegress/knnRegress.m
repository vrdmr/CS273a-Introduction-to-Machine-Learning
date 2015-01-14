function obj = knnRegress(Xtr,Ytr,K,alpha)
% knnRegress(X,Y,K,alpha) : construct a k-nearest-neighbor regression model
% takes no arguments, or training data and K
% alpha: weighted average coefficient (Gaussian weighting); alpha=0 => simple average
  if (nargin < 3) K = 1; end;
  if (nargin < 4) alpha = 0; end;
  obj.K=K; obj.Xtrain=[]; obj.Ytrain=[]; obj.alpha=alpha;
  obj=class(obj,'knnRegress');
	if (nargin > 0) 
		obj.K = K;
		obj.Xtrain = Xtr;
		obj.Ytrain = Ytr;
		obj.alpha = alpha;
	end;
end

