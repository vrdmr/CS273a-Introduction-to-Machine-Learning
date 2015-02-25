% err = loglikelihood(obj, X,Y) : compute the empirical avg log likelihood of "obj" on test data (X,Y)
function e = loglikelihood(obj,Xte,Yte)             
  e = mean( sum( log(predictSoft(obj,Xte).^Yte) ,2) ,1);  
end
