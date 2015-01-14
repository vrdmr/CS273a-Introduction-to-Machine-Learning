% Train random forest regression tree
% function obj=train(obj, data,target, varargin)
% 
function obj=train(obj, data,target, Nmin,DepthMax,VarMin,nFeat)
  [N,D] = size(data);
  if (nargin < 4) Nmin = 10; else Nmin = max(Nmin,2); end;
  if (nargin < 5) DepthMax=10; end; 
  if (nargin < 6) VarMin=.001; end;
  if (nargin < 7) nFeat = round(D/2); else nFeat = min(D,nFeat); end;

  % Allocate memory for binary tree with given max depth and >= 1 datum per leaf
  sz = min(2*N,2^(DepthMax+1)); 
  L=zeros(1,sz); R=L; F=L; T=L;

  % Unfortunately, Matlab cannot use the JIT compiler if we pass an object, so pass arrays
  [L,R,F,T, last] = dectreeTrain(data,target, L,R,F,T, 1, 0,  Nmin,DepthMax,VarMin,nFeat);

  % Store returned data structure info in the object
  obj.L=L(1:last-1); obj.R=R(1:last-1); obj.F=F(1:last-1); obj.T=T(1:last-1);



%%%%%% Recursive training function:

function  [L,R,F,T, next] = dectreeTrain(data,target,  L,R,F,T,  next, depth, Nmin,DepthMax,VarMin,nFeat)

[N,D] = size(data);

% check leaf conditions:
if (N<Nmin || depth >= DepthMax || var(target)<VarMin )
  if (N==0) error('Tried to create size-zero node'); end;
  F(next) = 0;
  T(next) = mean(target);
  next = next+1;
  return;
end;

% otherwise, search over (allowed) features
BestVal = inf;
BestFeat = -1;
TryFeat = randperm(D);
for iFeat = TryFeat(1:nFeat),

  [dsorted,pi] = sort(data(:,iFeat)');                % sort data, targets by feature id
  tsorted = target(pi)';
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
  T(next) = mean(target);
  next = next+1;
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


