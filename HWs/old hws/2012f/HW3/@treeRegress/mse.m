function e = mse(obj,Xte,Yte)             % calculate the mse error
  e = mean( (Yte - predict(obj,Xte)).^2 );  % for the full ensemble
end
