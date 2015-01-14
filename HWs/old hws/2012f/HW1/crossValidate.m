function [Xtr Xte Ytr Yte] = crossValidate(X,Y,nFolds,iFold)
% split data for n-fold cross validation
% [xTrain xTest yTrain yTest] = crossValidat(X,Y,nFolds,iFold) splits the (X,Y) data into the i-th of n folds
  [Nx Mx] = size(X);
  N  = fix(Nx/nFolds);
  if (~isempty(Y))
    [Ny My] = size(Y);
    if (Nx~=Ny) error('X and Y should be the same number of rows'); end;
  end;
  if (iFold == nFolds) 
    Xte = X( (iFold-1)*N+1:Nx, :);  
    Xtr = X( [1:(iFold-1)*N] , :);  
    if (~isempty(Y))
      Yte = Y( (iFold-1)*N+1:Ny, :);  
      Ytr = Y( [1:(iFold-1)*N] , :);  
    end;
  else
    Xte = X( (iFold-1)*N+1:iFold*N , :);  
    Xtr = X( [1:(iFold-1)*N,iFold*N+1:Nx] , :);  
    if (~isempty(Y))
      Yte = Y( (iFold-1)*N+1:iFold*N , :);  
      Ytr = Y( [1:(iFold-1)*N,iFold*N+1:Ny] , :);  
    end;
  end;
  

