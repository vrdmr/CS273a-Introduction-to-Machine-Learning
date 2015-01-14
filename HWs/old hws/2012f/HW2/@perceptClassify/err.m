function e = err(obj, Xte, Yte)
  Yhat = predict(obj, Xte);
  e = mean( Yhat ~= Yte );
end
