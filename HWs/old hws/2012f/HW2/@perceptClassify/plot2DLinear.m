function plot2DLinear(obj, Xtrain, Ytrain)
% plot2DLinear(obj, Xtrain,Ytrain)
%   plot a linear classifier (data and decision boundary) when features Xtrain are 2-dim
%   wts are 1x3,  wts(0)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(Xtrain);
  if (d~=2) error('Sorry -- plot2DLinear only works on 2D data...'); end;
  c0 = find(Ytrain==-1); c1=find(Ytrain==1);
  Xplt = linspace(min(Xtrain(:,1)),max(Xtrain(:,1)),200);
  % decision boundary is:
  %  sign( w2 x2 + w1 x1 + w0 ) = 0  <=> w2 x2 + w1 x1 + w0 = 0  <=> x2 = -w0 -w1/w2 x1;
  plot(Xtrain(c0,1),Xtrain(c0,2),'bo',...       % class zero training data
       Xtrain(c1,1),Xtrain(c1,2),'gs',...       % class one training data
       Xplt,-obj.wts(1)/obj.wts(3) - obj.wts(2)/obj.wts(3).*Xplt,'r-');  % decision boundary
  drawnow;  
  
