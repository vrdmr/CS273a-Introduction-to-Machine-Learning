function plot1DLogistic(obj, Xtrain, Ytrain)
% plot1DLogistic(obj, Xtrain,Ytrain) 
%  plot a logistic classifier when training features Xtrain are univariate
%  wts = length-2 vector; yhat = logit(wts(1) + wts(2)*X)
%
  [n,d] = size(Xtrain);
  if (d~=1) error('Sorry -- plot1DLogistic only works on 1D data...'); end;
  xplt = linspace(min(Xtrain), max(Xtrain), 200)';
  c0 = find(Ytrain==0); c1=find(Ytrain==1);
  fig(2); plot(Xtrain(c0),Ytrain(c0),'bo',Xtrain(c1),Ytrain(c1),'rs', ...
     xplt,obj.wts(1)+obj.wts(2:end)*xplt,'g-',  xplt,logistic(obj,xplt),'k-');
