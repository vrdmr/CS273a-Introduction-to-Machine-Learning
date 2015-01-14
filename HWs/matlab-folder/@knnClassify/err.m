function e = err(obj, Xte, Yte)
% err(obj, X,Y)  : compute error rate on test data (X,Y)

  Yhat = predict(obj, Xte);
  e = mean( Yhat ~= Yte );

