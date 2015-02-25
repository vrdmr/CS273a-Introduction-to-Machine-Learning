% e = errK(obj, Xtest, Ytest) : compute misclassification error; assumes Ytest is 1-of-K
function e = errK(obj, Xte, Yte)
  Yhat = predict(obj, Xte);
  e = mean( Yhat ~= from1ofK(Yte,obj.classes) );
end
