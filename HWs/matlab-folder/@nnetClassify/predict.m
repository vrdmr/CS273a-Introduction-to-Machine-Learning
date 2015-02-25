function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

  Rte = predictSoft(obj,Xte);  % compute soft output values
  [nil, Yte] = max(Rte,[],2);  % get index of maximum response
  Yte = obj.classes(Yte);      % convert to saved class values

