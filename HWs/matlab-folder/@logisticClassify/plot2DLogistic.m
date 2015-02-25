function plot2DLogistic(obj, X, Y)
% plot2DLogistic(obj, X,Y)
%   plot a logistic classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(0)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  % Plot data using helper function
  plotClassify2D([],X,Y); hold on;

  cmap = jet(256);
  col = fix( (unique(Y) - min(Y))./(max(Y)-min(Y)) *(size(cmap,1)-1)+1);

  ax = axis(); 
  xs = linspace(ax(1),ax(2),200);

  % decision boundary is:
  %  logistic( w2 x2 + w1 x1 + w0 ) =.5  <=> w2 x2 + w1 x1 + w0 = 0  <=> x2 = -w0 -w1/w2 x1;
  wts0 = [zeros(1,d+1);obj.wts];   % augment with zero parameters
  C = length(obj.classes);
  %add = .01*(ax(2)-ax(1));         % will plot "a tiny bit" above/below decision boundary
  for i=1:C, for j=i+1:C,
    w = wts0(i,:) - wts0(j,:);     % get pairwise difference of weights to compare class i / j boundary
    plot(xs,-w(1)./(w(3)+eps) - w(2)./(w(3)+eps) * xs,'k-');  % decision boundary 
    %plot(xs,-w(1)./w(3) - w(2)./w(3) * (xs +add),'linewidth',3,'Color', cmap(col(i),:));  % decision boundary (pos)
    %plot(xs,-w(1)./w(3) - w(2)./w(3) * (xs -add),'linewidth',3,'Color', cmap(col(j),:));  % decision boundary (neg)
%xs, -w(1)./w(3) - w(2)./w(3) * xs,
%fprintf('.'); pause;

  end; end;
  axis(ax); 
  hold off;
  drawnow;  
  
