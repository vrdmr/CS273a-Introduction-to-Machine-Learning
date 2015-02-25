function C = confusion(obj, Xte, Yte)
% C = confusion(learner, Xtest, Ytest) : estimate confusion matrix (Y x Yhat) from test data
  Yhat = predict(obj, Xte);
  nClasses = length(obj.classes);
  idx = toIndex(Yte,obj.classes) + nClasses*(toIndex(Yhat,obj.classes)-1);
  C   = hist(idx,1:nClasses^2);
  C   = reshape(C, nClasses,nClasses);

