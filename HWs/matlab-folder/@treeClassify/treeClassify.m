function obj = treeClassify(Xtr,Ytr,varargin)
% treeClassify(Xtr,Ytr,...) : construct a decision tree classifier
% for arguments sett treeClassify/train

  % classifier is stored as a simple linear-memory binary tree
  obj.L=[0];   % index of left child
  obj.R=[0];   % index of right child
  obj.F=[0];   % feature to split on (0 = leaf = predict)
  obj.T=[0];   % threshold to split at (or prediction value if leaf)

  obj.classes = [];  % list of output class values

  obj=class(obj,'treeClassify');
  if (nargin > 0)
    obj=train(obj, Xtr,Ytr,varargin{:});
  end;
end

