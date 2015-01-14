    % Batch training: just memorize data
    function obj=train(obj, Xtr,Ytr)
      obj.Xtrain = Xtr;
      obj.Ytrain = Ytr;
    end
