function plotClassify2D(learner,X,Y,pre,N)
% plotClassify2D(learner,X,Y [,pre]) : plot data and classifier outputs on two-dimensional data
% This function plots the data (X,Y) and "predict(learner,X,Y)" together.
%    The learner is predicted on a dense grid covering the data X, to show its decision boundary
%
% Optional arguments: 
%    pre : function handle applied to X before predict, e.g., predict(learner,pre(X),Y)
% Ex: pre = @(x) fpoly(x,2);                         % applies a polynomial expansion before prediction
% Ex: [Xtr,M,S]=whiten(Xtr); pre=@(x) whiten(x,M,S); % applies whitening transform before predict

if (size(X,2)~=2) error('plotClassify2D must be called using two-dimensional data (features)'); end;

%forig = gcf;
%f=figure; 
plot(X(:,1),X(:,2),'k.');
ax=axis;  % get current plot appearance
if (nargin < 5) N=256; end;    % density of evaluation
%close(f);

% Evaluate each point of feature space and predict the class
X1 = linspace(ax(1),ax(2),N); X1sp=X1'*ones(1,N);
X2 = linspace(ax(3),ax(4),N); X2sp=ones(N,1)*X2;
Xfeat = [X1sp(:),X2sp(:)];

% preprocess / create feature vector if necessary
if (nargin > 3) Xfeat = pre(Xfeat); end;

% if no learner passed, just predict zero; otherwise use learner's predict function
if (isempty(learner)) pred=0*Xfeat(:,1);
else pred = predict(learner,Xfeat);
end;

% plot decision values for the space in "faded" color
cmap=jet(256); 
clim=unique(Y)';
if (length(clim)==1) clim=[clim clim+1]; end;
if (isempty(learner)) cmapshade = ones(256,3); else cmapshade = cmap*.4+.6; end;
%figure(gcf); 
imagesc(X1,X2,reshape(pred,N,N)',[clim(1) clim(end)]); axis xy; hold on; colormap(cmapshade);

% plot each classes' data in the full spectrum color
for c=clim,
  col= fix((c-min(clim))/(max(clim)-min(clim))*255+1);
  %H=plot(X(find(Y==c),1),X(find(Y==c),2),'ko','markersize',7,'markerfacecolor',cmap(col,:)); 
  H=plot(X(find(Y==c),1),X(find(Y==c),2),'o','markersize',7,'color',cmap(col,:),'markerfacecolor',cmap(col,:)); 
  %set(H,'markerfacecolor',cmap(col,:));
end;

