    % calculate mean squared error for a given validation data set
    function err = mse(obj,Xval,Yval)
      Yhat = predict(obj, Xval);
      err = mean( (Yhat-Yval).^2 );
    end

