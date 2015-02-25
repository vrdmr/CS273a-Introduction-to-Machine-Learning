function obj = knnClassify(Xtr,Ytr,K,alpha)
% knnClassify(X,Y,...) : create k-nearest-neighbors classifier
% takes no arguments, or training data to be used in constructing the classifier
% alpha: weighted average coefficient (Gaussian weighting); alpha=0 => simple average

  if (nargin < 3) K = 1; end;
  if (nargin < 4) alpha = 0; end;
  obj.K=K; obj.Xtrain=[]; obj.Ytrain=[]; obj.classes=[]; obj.alpha=alpha;
  obj=class(obj,'knnClassify');
  if (nargin > 0) 
    obj = train(obj,Xtr,Ytr);
    obj = setK(obj,K);
    obj.alpha = alpha;
  end;
end

