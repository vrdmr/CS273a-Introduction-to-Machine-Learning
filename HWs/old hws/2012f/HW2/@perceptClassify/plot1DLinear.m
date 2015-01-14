function plot1DLinear(obj, Xtrain, Ytrain)
% plot1DLinear(obj, Xtrain,Ytrain) 
%  plot a linear classifier when training features Xtrain are univariate
%  wts = length-2 vector; yhat = logit(wts(1) + wts(2)*X)
%
  [n,d] = size(Xtrain);
  if (d~=1) error('Sorry -- plot1DLinear only works on 1D data...'); end;
  xplt = linspace(min(Xtrain), max(Xtrain), 200)';
  xplt1 = [1+0*xplt xplt];
  c0 = find(Ytrain==-1); c1=find(Ytrain==1);
  plot(Xtrain(c0),Ytrain(c0),'bo',Xtrain(c1),Ytrain(c1),'gs',... % data colored by class
       xplt,min(1,max(-1,xplt1*obj.wts')),'r-');   % thresholded linear response value
