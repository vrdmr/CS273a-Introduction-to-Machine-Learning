function obj=train(obj, Xtr,Ytr)
% knn=train(knn,Xtrain,Ytrain) : Batch training for knn; just memorize data
  obj.Xtrain = Xtr;
  obj.Ytrain = Ytr;
  obj.classes= unique(Ytr);

