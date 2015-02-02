function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

wts = getWeights(obj);
yhat = zeros(size(Xte,1),1);
f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2 ;

for i=1:size(Xte,1);
    x = sign(f(Xte(i,1),Xte(i,2)));
    yhat(i) = obj.classes(ceil((x+3)/2));
end;

Yte = yhat;