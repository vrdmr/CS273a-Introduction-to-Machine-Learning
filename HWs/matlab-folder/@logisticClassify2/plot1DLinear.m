function plot1DLinear(obj, X, Y)
% plot1DLinear(obj, X,Y) 
%  plot a linear/logistic classifier when training features X are univariate
%  wts = length-2 vector; yhat = logit(wts(1) + wts(2)*X)
%
  [n,d] = size(X);
  if (d~=1) error('Sorry -- plot1DLinear only works on 1D data...'); end;
  xplt = linspace(min(X), max(X), 200)';
  c0 = find(Y==obj.classes(1)); c1=find(Y==obj.classes(2));
  fig(2); plot(X(c0),Y(c0),'bo',X(c1),Y(c1),'rs', ...
     xplt,obj.wts(1)+obj.wts(2:end)*xplt,'g-',  xplt,logistic(obj,xplt),'k-');
