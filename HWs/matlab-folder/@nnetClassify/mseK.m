% err = mseK(obj, X,Y) : compute the mean squared error of predictor; assumes Y is 1-of-K
function e = mseK(obj,Xte,Yte)             
  e = mean( sum( (Yte - predictSoft(obj,Xte)).^2 ,2) ,1);  
end
