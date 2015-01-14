    % calculate mean absolute error for a given validation data set
    function err = mae(obj,Xval,Yval)
      Yhat = predict(obj, Xval);
      err = mean( abs(Yhat-Yval) );
    end
