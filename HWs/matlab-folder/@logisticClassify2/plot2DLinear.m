function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;


%%% TODO: Fill in the rest of this function...
wts = getWeights(obj);
%forig = gcf;
f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2 ;

scatter(X((Y==0),1),X((Y==0),2),50,'or');
hold on;
scatter(X((Y==1),1),X((Y==1),2),50,'+b');
hold on;
%ARGS = [3,7,1.5,5]
ezplot(f,[4,7,1.5,5])

legend('Class 0','Class 1');

hold off;
%{

ax=axis;  % get current plot appearance
N=256;    % density of evaluation
%close(f);

% Evaluate each point of feature space and predict the class
X1 = linspace(ax(1),ax(2),N); X1sp=X1'*ones(1,N);
X2 = linspace(ax(3),ax(4),N); X2sp=ones(N,1)*X2;
Xfeat = [X1sp(:),X2sp(:)];

% preprocess / create feature vector if necessary
if (nargin > 3) Xfeat = pre(Xfeat); end;

% if no learner passed, just predict zero; otherwise use learner's predict function
% pred = predict(obj,Xfeat);

% plot decision values for the space in "faded" color
cmap=jet(256); 
clim=unique(Y)';
if (length(clim)==1) clim=[clim clim+1]; end;
if (isempty(obj)) cmapshade = ones(256,3); else cmapshade = cmap*.4+.6; end;
%figure(gcf); 
%imagesc(X1,X2,reshape(pred,N,N)',[clim(1) clim(end)]); axis xy; hold on; colormap(cmapshade);

% plot each classes' data in the full spectrum color
for c=clim,
  col= fix((c-min(clim))/(max(clim)-min(clim))*255+1);
  H=plot(X(find(Y==c),1),X(find(Y==c),2),'o','markersize',7,'color',cmap(col,:),'markerfacecolor',cmap(col,:)); 
end;
%}
