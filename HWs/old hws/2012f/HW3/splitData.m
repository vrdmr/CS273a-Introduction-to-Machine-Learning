function [Xtr Xte Ytr Yte] = splitData(X, Y, trainFraction)
% split data into training and test sets
% [xTrain xTest yTrain yTest] = splitData(X,Y,trainFraction) splits the (X,Y) data into training and test sets
%   where the training data are the first (trainFraction) part of the data.
% see also: permuteData, bootstrapData, crossValidate
  [Nx,Mx] = size(X);
  Ne = round(trainFraction*Nx);
  Xtr = X(1:Ne,:); Xte=X(Ne+1:end,:);
  if (~isempty(Y))
    [Ny,My] = size(Y);
    if (Nx ~= Ny) error('X and Y should have the same number of rows'); end;
    Ytr = Y(1:Ne,:); Yte=Y(Ne+1:end,:);
  end;

  

