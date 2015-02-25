% err = mse(obj, X,Y) : compute the mean squared error of predictor "obj" on test data (X,Y)
function e = mse(obj,Xte,Yte)             
  e = mseK(obj,Xte,to1ofK(Yte));
end
