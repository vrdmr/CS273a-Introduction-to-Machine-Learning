    % Batch, closed form solution
    function obj=train(obj, Xtr,Ytr, reg)
      if (nargin < 4)
        obj.theta = (Xtr\Ytr)';
      else
        [M,N]=size(Xtr);
        obj.theta = Ytr'*(Xtr/M)*inv(Xtr'*Xtr/M + reg*eye(N));
      end;
    end

