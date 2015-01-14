    % Constructor (takes no arguments or training data)
    function obj = linearRegress(Xtr,Ytr, varargin)
      obj.theta=[];
      obj=class(obj,'linearRegress');
      if (nargin > 0) 
        obj=train(obj,Xtr,Ytr, varargin{:});
      end;
    end

