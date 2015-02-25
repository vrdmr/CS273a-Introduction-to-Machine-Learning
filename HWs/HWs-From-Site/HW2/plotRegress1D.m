function plotRegress1D(learner,X,Y,pre,Color)
% plotRegress1D(learner,X,Y [,pre,color]) : plot data and regression output on one-dimensional data
% This function plots the data (X,Y) and "predict(learner,X,Y)" together.
%    pre : function handle applied to X before predict, e.g., predict(learner,pre(X),Y)
% Ex: pre = @(x) fpoly(x,2);       % applies a polynomial expansion before prediction
% Ex: [Xtr,M,S]=whiten(Xtr); pre=@(x) whiten(x,M,S);   % applies whitening transform before predict

fuse = gcf;
f=figure; plot(X(:,1),Y,'k.');
ax=axis;  % get current plot appearance
N=256;
close(f);
fig(gcf); holds = ishold;

X1 = linspace(ax(1),ax(2),N)'; 

% preprocess / create feature vector if necessary
if (nargin > 3) Xfeat = pre(X1); else Xfeat = X1; end;

% if no learner passed, just predict zero; otherwise use learner's predict function
if (isempty(learner)) pred=[]; %0*Xfeat(:,1);
else pred = predict(learner,Xfeat);
end;

if (nargin < 5) Color = [1 .2 .2]; end;
%figure; 
if (~isempty(pred)) plot(X1,pred,'k-','linewidth',3); hold on; end;
plot(X(:,1),Y,'o','markersize',7,'color',Color,'markerfacecolor',Color);
axis(ax);
if (holds) hold on; else hold off; end;


