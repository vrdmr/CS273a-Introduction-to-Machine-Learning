function e = nll(obj, Xte, Yte)
% compute logistic negative log-likelihood loss
  Yte = logical(to1ofK(Yte,obj.classes));
  Yhat = predictSoft(obj, Xte);
  e = - mean( log(Yhat(Yte)) );

  %Y( Y==obj.classes(1) )=0;            % and convert to canonical 0/1 class values
  %Y( Y==obj.classes(2) )=1;
  %e = - mean( [log(Yhat(Yte==2)) ; log(1-Yhat(Yte==1))] );  % compute neg log likelihood

