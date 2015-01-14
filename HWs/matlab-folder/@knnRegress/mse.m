function e = mse(obj,Xte,Yte)             
% err = mse(obj, X,Y) : compute the mean squared error of predictor "obj" on test data (X,Y)
  e = mean( sum( (Yte - predict(obj,Xte)).^2,2) );   % sum(*,2) in case of multivariate y

