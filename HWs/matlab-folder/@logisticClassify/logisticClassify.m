function obj = logisticClassify(Xtr,Ytr, varargin)
% logisticClassify(X,Y,...) : construct a logistic classifier (linear classifier with saturated output)
% can take no arguments, or see logisticClassify/train for training options

  obj.wts=[];         % linear weights on features (1st weight is constant term)
  obj.classes=[];     % list of class values used in input
  obj=class(obj,'logisticClassify');
  if (nargin > 0) 
    obj=train(obj,Xtr,Ytr, varargin{:});
  end;
end

