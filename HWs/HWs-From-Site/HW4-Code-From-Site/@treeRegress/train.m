function obj=train(obj, X,Y, varargin)
% Train random forest classification tree
% obj=train(obj, X,Y, ...)
% Optional args:
%   'minParent',<int>: minimum # of data required to split a node
%   'maxDepth',<int>:  maximum depth of the decision tree
%   'minScore',<dbl>:  minimum value of the score improvement to split a node
%   'nFeatures',<int>: # of available features for splitting at each node

  [N,D] = size(X);

  % default options:
  minParent = 2;
  maxDepth = inf;
  minScore = -1;
  nFeat = D;

  i=1;                                       % parse through various options
  while (i<=length(varargin)),
    switch(lower(varargin{i}))
    case 'weights',   error('Weights not supported yet.'); %wts = varargin{i+1}; i=i+1;
    case 'minparent', minParent = max(varargin{i+1},2); i=i+1;
    case 'maxdepth',  maxDepth = varargin{i+1}; i=i+1;
    case 'minscore',  minScore = varargin{i+1}; i=i+1;
    case 'nfeatures', nFeat = varargin{i+1}; i=i+1;
    end;
    i=i+1;
  end;
  nFeat = min(D,nFeat);

  % Allocate memory for binary tree with given max depth and >= 1 datum per leaf
  sz = min(2*N,2^(maxDepth+1)); 
  L=zeros(1,sz); R=L; F=L; T=L;

  % Unfortunately, Matlab cannot use the JIT compiler if we pass an object, so pass arrays
  [L,R,F,T, last] = dectreeTrain(X,Y, L,R,F,T, 1, 0,  minParent,maxDepth,minScore,nFeat);

  % Store returned data structure info in the object
  obj.L=L(1:last-1); obj.R=R(1:last-1); obj.F=F(1:last-1); obj.T=T(1:last-1);



%%%%%% Recursive training function:

function  [L,R,F,T, next] = dectreeTrain(X,Y,  L,R,F,T,  next, depth, minParent,maxDepth,minScore,nFeat)

[N,D] = size(X);

% check leaf conditions:
if (N<minParent || depth >= maxDepth || var(Y)<minScore )
  if (N==0) error('Tried to create size-zero node'); end;
  F(next) = 0;
  T(next) = mean(Y,1);
  next = next+1;
  return;
end;

% otherwise, search over (allowed) features
BestVal = inf;
BestFeat = -1;
TryFeat = randperm(D);
for iFeat = TryFeat(1:nFeat),

  [dsorted,pi] = sort(X(:,iFeat)');                   % sort data, targets by feature id
  tsorted = Y(pi)';
  % TODO: should compare for numerical tolerance
  can_split = [dsorted(1:end-1)~=dsorted(2:end)  0];  % which indices are valid split points?
  if (~any(can_split)) continue; end;
  
  % compute mean up to position j, and mean past position j, for all j
  ycumTo = cumsum(tsorted);
  ycumPa = ycumTo(end) - ycumTo;
  meanTo = ycumTo ./ (1:N);
  meanPa = ycumPa ./ [N-1:-1:1 1];
  % compute variance up to position j, and past position j, for all j
  y2cumTo= cumsum(tsorted.^2);
  y2cumPa= y2cumTo(end) - y2cumTo;
  varTo = (y2cumTo - 2.*meanTo.*ycumTo + (1:N).*meanTo.^2) ./ (1:N);
  varPa = (y2cumPa - 2.*meanPa.*ycumPa + (N-1:-1:0).*meanPa.^2) ./ [N-1:-1:1 1];
  varPa(end)=inf;
  
  % find minimum weighted variance among all split points
  weightedVariance = ( (1:N)/N.*varTo + (N-1:-1:0)/N.*varPa );
  [val,idx] = min( (weightedVariance + 1) ./ (can_split+1e-100) );  % find only splittable points
  
  % save best feature, split point found so far 
  if (val < BestVal)
    BestVal = val;
    BestFeat = iFeat;
    BestThresh = (dsorted(idx)+dsorted(idx+1))/2;
  end;
  
end;

if (BestFeat == -1)  % no split possible?  output a leaf node
  F(next) = 0;
  T(next) = mean(Y);
  next = next+1;
  return;
end;



%%%%%%%
% split data on feature iFeat, value (tsorted(idx)+tsorted(idx+1))/2
F(next) = BestFeat;
T(next) = BestThresh;
goLeft = X(:,F(next)) < T(next);
myidx = next;
next = next+1;

 
L(myidx) = next;   % Recurse left
[L,R,F,T,next]=dectreeTrain(X(goLeft,:),Y(goLeft), L,R,F,T, next,depth+1, minParent,maxDepth,minScore,nFeat);

R(myidx) = next;   % Recurse right
[L,R,F,T,next]=dectreeTrain(X(~goLeft,:),Y(~goLeft), L,R,F,T, next,depth+1, minParent,maxDepth,minScore,nFeat);


