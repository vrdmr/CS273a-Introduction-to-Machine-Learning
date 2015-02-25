function obj = linearRegress(Xtr,Ytr, varargin)
% linearRegress(X,Y,...) : construct linear regression model 
% takes no arguments, or training parameters; see linearRegress/train
  obj.theta=[];
  obj=class(obj,'linearRegress');
  if (nargin > 0) 
    obj=train(obj,Xtr,Ytr, varargin{:});
  end;
end

