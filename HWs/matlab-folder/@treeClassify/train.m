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
    case 'minscore',  error('MinScore not supported'); %minScore = varargin{i+1}; i=i+1;
    case 'nfeatures', nFeat = varargin{i+1}; i=i+1;
    end;
    i=i+1;
  end;
  nFeat = min(D,nFeat);

  % Get class id values and replace with values 1..C  
  if (isempty(obj.classes))
    obj.classes = unique(Y);
  end 
  Y = toIndex(Y,obj.classes);
  %Y2 = Y; for i=1:length(obj.classes) Y2(Y==obj.classes(i))=i; end; Y=Y2;
  
  % Allocate memory for binary tree with given max depth and >= 1 datum per leaf
  sz = min(2*N,2^(maxDepth+1)); 
  L=zeros(1,sz); R=L; F=L; T=L;
  
  % Unfortunately, Matlab cannot use the JIT compiler if we pass an object, so pass arrays
  [L,R,F,T, last] = dectreeTrain(X,Y, L,R,F,T, 1, 0,  minParent,maxDepth,minScore,nFeat);
  
  % Store returned data structure info in the object
  obj.L=L(1:last-1); obj.R=R(1:last-1); obj.F=F(1:last-1); obj.T=T(1:last-1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Recursive training function:

function  [L,R,F,T, next] = dectreeTrain(X,Y,  L,R,F,T,  next, depth, minParent,maxDepth,minScore,nFeat)

[N,D] = size(X);             % get data (sub)set size
nClasses = max(Y);           % assumes classes 1..C

% check leaf conditions:
if (N<minParent || depth >= maxDepth || all(Y==Y(1)))
  if (N==0) error('Tried to create size-zero node'); end;
  F(next) = 0;               % mark node as a leaf
  tmp = sum( to1ofK(Y, 1:nClasses) ,1);    % compute # of data in each class 
  [nc,T(next)] = max(tmp);   % find majority vote
  next = next+1;             % advance to next node id
  return;
end;

% otherwise, search over (allowed) features
BestVal = -inf;  
BestFeat = -1;
TryFeat = randperm(D);
for iFeat = TryFeat(1:nFeat),

  [dsorted,pi] = sort(X(:,iFeat)');                  % sort data, targets by feature id
  tsorted = Y(pi);
  % TODO: should compare for numerical tolerance
  can_split = [dsorted(1:end-1)~=dsorted(2:end)  0]; % which indices are valid split points?
  if (~any(can_split)) continue; end;
  
  % compute class probabilities left of position j, and right of j, for all j
  yLeft = cumsum( to1ofK(tsorted, 1:nClasses) );    % compute # of data in each class cumulatively from small->large
  yRight = bsxfun(@minus, yLeft(end,:), yLeft);     %  & similarly cumulatively from large -> small
  yLeft  = bsxfun(@rdivide, yLeft, [1:N]');         %  convert to the fraction of data in each class
  yRight = bsxfun(@rdivide, yRight, [N-1:-1:1 1]'); %   for each split (To,Pa)

  % Compute entropy for score function
  Hroot = -yLeft(end,:) * log(yLeft(end,:)'+eps);
  Hleft = -sum(yLeft.*log(yLeft+eps),2);
  Hright= -sum(yRight.*log(yRight+eps),2); 
                        
  % find maximum information gain among all split points
  IG = Hroot - ( (1:N)/N.*Hleft' + (N-1:-1:0)/N.*Hright' );
  [val,idx] = max( (IG+eps) .* can_split );        % find only splittable points

  % save best feature, split point found so far 
  if (val > BestVal)
    BestVal = val;
    BestFeat = iFeat;
    BestThresh = (dsorted(idx)+dsorted(idx+1))/2;
  end;
  
end;

if (BestFeat == -1)  % no split possible?  output a leaf node
  F(next) = 0;               % mark node as a leaf
  tmp = sum( to1ofK(Y, 1:nClasses) );    % compute # of data in each class 
  [nc,T(next)] = max(tmp);   % find majority vote
  next = next+1;             % advance to next node id
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REPLACEMENT / INTERNAL HELPER to1ofK : specialized to already use indexed Y-values
function Yext = to1ofK(Y,C)
  [N,D] = size(Y);
  Yext = zeros(N,C(end)); 
  Yext( (1:N) + N*(Y'-1))=1;


