function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

  [nil,pred] = max(logistic(obj,Xte),[],2);
  Yte = obj.classes( pred );

