function Yte = predict(obj,Xte)
% Yhat = predict(obj,Xtest) : make predictions on test data with the classifier

  p = predictSoft(obj,Xte);
  [tmp,c] = max(p,[],2);                   % find the index of the largest probability
  Yte = obj.classes(c);                    % and return that class ID


