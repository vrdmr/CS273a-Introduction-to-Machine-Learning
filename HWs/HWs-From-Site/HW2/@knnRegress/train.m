function obj=train(obj, Xtr,Ytr)
% knn=train(knn,Xtrain,Ytrain) : Batch training of knn learner; just memorize data
  obj.Xtrain = Xtr;
  obj.Ytrain = Ytr;
end
