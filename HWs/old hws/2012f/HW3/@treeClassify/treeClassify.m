function obj = treeClassify(Xtr,Ytr,varargin)
% treeClassify(Xtr,Ytr,...) : construct a decision tree classifier
% for arguments sett treeClassify/train
  %[N,D] = size(Xtr);
  %obj.L=zeros(1,N);  % left child
  %obj.R=zeros(1,N);  % right child
  %obj.F=zeros(1,N);  % feature to split on (0 = predict)
  %obj.T=zeros(1,N);  % threshold to split at (or prediction value)
  obj.L=[0]; obj.R=[0]; obj.F=[0]; obj.T=[0];
  obj.classes = [];

  obj=class(obj,'treeClassify');
  if (nargin > 0)
    obj=train(obj, Xtr,Ytr,varargin{:});
  end;
end

