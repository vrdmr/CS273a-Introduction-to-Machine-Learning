function obj = treeRegress(Xtr,Ytr,varargin)
% treeRegress(Xtr,Ytr,...) : construct a decision tree regression model
% can take no arguments, or see treeRegress/train for training arguments

  %[N,D] = size(Xtr);
  %obj.L=zeros(1,N);  % left child
  %obj.R=zeros(1,N);  % right child
  %obj.F=zeros(1,N);  % feature to split on (0 = predict)
  %obj.T=zeros(1,N);  % threshold to split at (or prediction value)
  obj.L=[0]; obj.R=[0]; obj.F=[0]; obj.T=[0];

  obj=class(obj,'treeRegress');
  if (nargin > 0)
    obj=train(obj, Xtr,Ytr,varargin{:});
  end;
end

