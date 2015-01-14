function obj=train(obj, data,target, Nmin,DepthMax,VarMin,nFeat)
% Train random forest regression tree
% obj=train(obj, X,Y [,minParent,maxDepth,minScore,nFeatures])
%   nParent:   minimum # of data required to split a node
%   maxDepth:  maximum depth of the decision tree
%   minScore:  minimum value of the score improvement to split a node
%   nFeatures: # of available features for splitting at each node

  [N,D] = size(data);
  if (nargin < 4) Nmin = 10; else Nmin = max(Nmin,2); end;
  if (nargin < 5) DepthMax=10; end; 
  if (nargin < 6) VarMin=.001; end;
  if (nargin < 7) nFeat = round(D/2); else nFeat=min(nFeat,D); end;

  % Get class id values and replace with values 1..C  
  obj.classes = unique(target);
  Y = target; for i=1:length(obj.classes) Y(target==obj.classes(i))=i; end; target=Y;
  
  % Allocate memory for binary tree with given max depth and >= 1 datum per leaf
  sz = min(2*N,2^(DepthMax+1)); 
  L=zeros(1,sz); R=L; F=L; T=L;
  
  % Unfortunately, Matlab cannot use the JIT compiler if we pass an object, so pass arrays
  [L,R,F,T, last] = dectreeTrain(data,target, L,R,F,T, 1, 0,  Nmin,DepthMax,VarMin,nFeat);
  
  % Store returned data structure info in the object
  obj.L=L(1:last-1); obj.R=R(1:last-1); obj.F=F(1:last-1); obj.T=T(1:last-1);


%%%%%% Recursive training function:

function  [L,R,F,T, next] = dectreeTrain(data,target,  L,R,F,T,  next, depth, Nmin,DepthMax,VarMin,nFeat)

[N,D] = size(data);       % get data (sub)set size
nClasses = max(target);   % assumes classes 1..C

% check leaf conditions:
if (N<Nmin || depth >= DepthMax || all(target==target(1)))
  if (N==0) error('Tried to create size-zero node'); end;
  F(next) = 0;               % mark node as a leaf
  tmp = zeros(nClasses,N); tmp( target-1+[1:nClasses:nClasses*N]')=1; tmp=sum(tmp,2);
  [nc,T(next)] = max(tmp);   % find majority vote
  next = next+1;             % advance to next node id
  return;
end;

% otherwise, search over (allowed) features
BestVal = -inf;  
BestFeat = -1;
TryFeat = randperm(D);
for iFeat = TryFeat(1:nFeat),

  [dsorted,pi] = sort(data(:,iFeat)');                % sort data, targets by feature id
  tsorted = target(pi)';
  can_split = [dsorted(1:end-1)~=dsorted(2:end)  0];  % which indices are valid split points?
  if (~any(can_split)) continue; end;
  
  % compute class probabilities left of position j, and right of j, for all j
  yLeft = zeros(nClasses,N); yLeft( tsorted-1+[1:nClasses:nClasses*N])=1;
  yLeft = cumsum(yLeft,2);
  yRight = bsxfun(@minus, yLeft(:,end), yLeft);                      
  yLeft  = bsxfun(@rdivide, yLeft, 1:N);           % compute fraction of data in each class
  yRight = bsxfun(@rdivide, yRight, [N-1:-1:1 1]); %  in each split (To,Pa)
  
  % Compute entropy for score function
  ep = 1e-15;
  Hroot = -yLeft(:,end)' * log(yLeft(:,end)+ep);
  Hleft = -sum(yLeft.*log(yLeft+ep),1);
  Hright= -sum(yRight.*log(yRight+ep),1); 
                        
  % find maximum information gain among all split points
  IG = Hroot - ( (1:N)/N.*Hleft + (N-1:-1:0)/N.*Hright );
  [val,idx] = max( (IG+ep) .* can_split );  % find only splittable points

  % save best feature, split point found so far 
  if (val > BestVal)
    BestVal = val;
    BestFeat = iFeat;
    BestThresh = (dsorted(idx)+dsorted(idx+1))/2;
  end;
  
end;

if (BestFeat == -1)  % no split possible?  output a leaf node
  F(next) = 0;               % mark node as a leaf
  tmp = zeros(nClasses,N); tmp( target-1+[1:nClasses:nClasses*N]')=1; tmp=sum(tmp,2);
  [nc,T(next)] = max(tmp);   % find majority vote
  next = next+1;             % advance to next node id
  return;
end;

%%%%%%%
% split data on feature iFeat, value (tsorted(idx)+tsorted(idx+1))/2
F(next) = BestFeat;
T(next) = BestThresh;
goLeft = data(:,F(next)) < T(next);
myidx = next;
next = next+1;

 
L(myidx) = next;   % Recurse left
[L,R,F,T,next]=dectreeTrain(data(goLeft,:),target(goLeft), L,R,F,T, next,depth+1, Nmin,DepthMax,VarMin,nFeat);

R(myidx) = next;   % Recurse right
[L,R,F,T,next]=dectreeTrain(data(~goLeft,:),target(~goLeft), L,R,F,T, next,depth+1, Nmin,DepthMax,VarMin,nFeat);


