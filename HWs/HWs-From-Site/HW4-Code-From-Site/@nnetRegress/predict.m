function Yte = predict(obj,Xte)
% Y=predict(nnet,Xtest)  : prediction of nnet on test data Xtest

  L = length(obj.wts);
  Zte = [ones(size(Xte,1),1) Xte];  % initialize to input features + constant
  for l=1:L-1
    Zte = Zte*obj.wts{l}';          % compute linear response of next layer
    Zte = [ones(size(Zte,1),1) obj.Sig(Zte)]; % activation f'n + constant
  end;
  Zte = Zte*obj.wts{L}';            % compute output layer linear response
  Yte = obj.Sig0(Zte);              % Output layer activation function

