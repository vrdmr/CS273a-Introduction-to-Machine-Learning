function [z,c,sumd] = kmeans(X,K,cInit,nIter)
% [assign, clusters, sumd] = kmeans(X, K [,initialClusters,nIter])
% Perform K-means clustering on data X (# data x # features)
%   initial clusters can be [K x N], or a method {'random', 'farthest', 'k++'}
%     random  : K random data points (uniformly) as clusters; 
%     farthest: choose cluster 1 uniformly, then the point farthest from all clusters so far, etc.
%     k++     : choose cluster 1 uniformly, then points randomly proportional to distance from current clusters
% Returns: 
%   assign (# data x 1) index of cluster
%   clusters (K x # features) cluster centers
%   sumd   (scalar) sum of squared euclidean distances
% Returns after nIter iterations (default 100) or when converged
%
% To convert clusters into a "clustering rule" that can be applied to future data, use the knn methods:
%   crule = knnClassify( clusters, (1:K)', 1 );  z = predict( crule, X );

if (nargin < 4) nIter=100;      end;
if (nargin < 3) cInit='random'; end;

doPlot = false;
%doPlot = true;

[N,D] = size(X);                             % get data size

%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%
if (isstr(cInit)), switch lower(cInit),      % if initialization passed in as a string
  case 'random',
    pi = randperm(N);
    c  = X(pi(1:K),:);
  case 'farthest',
    c  = kInit(X,K, true);
  case 'k++',
    c  = kInit(X,K, false);
  otherwise, error('Unknown initialization method');
  end;
else
  c  = cInit;
  if (isempty(K)) K=size(cInit,1); end;
end;
z = zeros(N,1);

%%%%%%%%%%%%%%%%%
%% Optimization
%%%%%%%%%%%%%%%%%
iter = 1;
done = (iter>nIter);
sumd = inf;  
sumdOld = inf;

while (~done) 
   sumd = 0;
   for i=1:N,
     dists = sum( (c - repmat(X(i,:),[K,1])).^2 , 2);  % compute distances from each cluster center
     [val,z(i)] = min(dists);                          % and assign datum i to the nearest cluster
     sumd = sumd + val;
   end

   for j=1:K,                                     % now update each cluster center j
     if (any(z==j)) c(j,:) = mean(X(z==j,:),1);   % to be the mean of the assigned data
     else c(j,:) = X(ceil(rand),:);               % (or random restart if no assigned data)
     end
   end

   if (doPlot && D==2)                            % if desired, plot for 2D data
     fig(1); plotClassify2D([],X,z); 
     hold on; plot(c(:,1),c(:,2),'kx','markersize',14,'linewidth',3); hold off; 
   end;

   done = (iter >= nIter) || (sumd == sumdOld);
   sumdOld = sumd;
   iter = iter+1;
end;

if (iter >= nIter) warning('kmeans:iter','kmeans:iter :: stopped after reaching maximum number of iterations'); end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% distance-based initialization:  randomly choose a point to start, then:
%  determ=1: choose the point farthest from the clusters chosen so far
%  determ=0: randomly choose new points proportionally to their distance
function clusters = kInit(X,K,determ)
  [m,n] = size(X);
  clusters = zeros(K,n);
  clusters(1,:) = X( ceil(rand*m), :);     % take a random point as the 1st cluster
  dist = sum( (X - ones(m,1)*clusters(1,:)).^2 , 2);
  for i=2:K,
    if (determ) [nil,j] = max( dist );     % choose farthest point
    else 
      pr = cumsum(dist); pr=pr/pr(end);    % or choose a random point by distance
      j = find( rand < pr , 1, 'first');
    end;
    clusters(i,:) = X(j,:);                % make that the next cluster & update min distances
    dist = min(dist, sum( (X - ones(m,1)*clusters(i,:)).^2 , 2) );
  end;


