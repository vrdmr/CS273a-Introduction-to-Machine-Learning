function Yte = predict(obj,Xte)
% Y = predict(tree, X) : make predictions on data X 
  Yte = dectreeTest(Xte, obj.L,obj.R,obj.F,obj.T, 1); 

function yhat = dectreeTest(data, L,R,F,T, pos) 
  yhat=zeros(size(data,1),1);
  if (F(pos)==0) yhat(:)=T(pos);
  else 
    goLeft = data(:,F(pos)) < T(pos);
    yhat(goLeft)  = dectreeTest(data(goLeft,:), L,R,F,T, L(pos)); 
    yhat(~goLeft) = dectreeTest(data(~goLeft,:), L,R,F,T, R(pos));
  end;
