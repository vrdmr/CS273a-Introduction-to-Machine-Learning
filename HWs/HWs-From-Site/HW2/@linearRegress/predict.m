function Yte = predict(obj,Xte)
% Yhat = predict(learner, Xtest) : make a prediction using the learned linear regression coefficients

  Xte = [ones(size(Xte,1),1) Xte];    % extend features by including a constant feature
  Yte = Xte * obj.theta';

