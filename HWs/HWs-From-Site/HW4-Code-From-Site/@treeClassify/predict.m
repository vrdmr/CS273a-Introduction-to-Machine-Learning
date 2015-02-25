function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

  Yte = dectreeTest(Xte, obj.L,obj.R,obj.F,obj.T, 1); 
  Yte = obj.classes(Yte);

function yhat = dectreeTest(X, L,R,F,T, pos) 
  yhat=zeros(size(X,1),1);
  if (F(pos)==0) yhat(:)=T(pos);
  else 
    goLeft = X(:,F(pos)) < T(pos);
    yhat(goLeft)  = dectreeTest(X(goLeft,:), L,R,F,T, L(pos)); 
    yhat(~goLeft) = dectreeTest(X(~goLeft,:), L,R,F,T, R(pos));
  end;
