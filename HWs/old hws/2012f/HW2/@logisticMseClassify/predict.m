% Test function: predict on Xtest
function Yte = predict(obj,Xte)
  Yte = obj.classes( sign( obj.wts(1) + Xte*obj.wts(2:end)')/2 + 1.5 );
end

