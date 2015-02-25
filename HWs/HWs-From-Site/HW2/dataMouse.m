function [X Y] = dataMouse(sz)
% [X Y] = dataMouse(size)  : create a 2D data set of classes +1/-1, indicated through mouse clicks
%   +1 = right button, -1 = left button, center button when done;  size = markersize (default 12)

fprintf('Click with the mouse to place data points.  Middle button ends.');
figure(1); clf; axis([0 10 0 10]); hold on;
col = 'rkb'; sh='oxs'; fa='rkw';
if (nargin < 1) sz=[12 12 12];
elseif (isscalar(sz)) sz=sz*[1 1 1]; 
end;
X=zeros(0,2); Y=zeros(0,1);

done = 0;
while (~done)
  [a b c] = ginput(1);

  done = (c==2);
  if (~done) 
    X = [X ; [a b]]; 
    Y=[Y;c];
    plot(X(end,1),X(end,2),[col(Y(end)) sh(Y(end))],'markersize',sz(Y(end)),'markerfacecolor',fa(Y(end))); 
  end;
end;

%X=X(:,1:end-1); Y=Y(1:end-1);
Y=Y-2;  % +1/-1


