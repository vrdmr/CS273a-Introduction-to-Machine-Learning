function err = mae(obj,Xval,Yval)
% err = mae(obj, X,Y) : compute the mean absolute error of predictor "obj" on test data (X,Y)

  err = mean( sum(abs(Yhat-predict(obj,Xval)),2) );  % sum(*,2) in case of multivariate y
