function [z,T,soft,ll] = emCluster(X,K,cInit,nIter,tol)
% [assign, clusters, soft, loglikelihd] = emCluster(X, K [,initialize,nIter,tol])
% Perform Gaussian mixture EM clustering on data X (# data x # features)
%   initial clusters can be [K x N], or a method {'random', 'farthest', 'k++'}
%     random  : K random data points (uniformly) as clusters; 
%     farthest: choose cluster 1 uniformly, then the point farthest from all clusters so far, etc.
%     k++     : choose cluster 1 uniformly, then points randomly proportional to distance from current clusters
% Returns: 
%   assign (# data x 1) index of cluster
%   clusters (struct): wt (Kx1), mu (K x # features), sig(#feat,#nfeat,K) : Gaussian component parameters
%   soft (#data x K) : soft assignment probabilities (rounded for assign)
%   loglikelihd (scalar) log-likelihood of the data under the returned model
% Returns after nIter iterations (default 100) or when converged to tolerance tol (default 1e-6)
%
% To convert clusters into a "clustering rule" that can be applied to future data, use the Gaussian Bayes methods:
%   (TODO)

reg = 1e-4;
if (nargin < 5) tol=1e-6;       end;
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
    perm = randperm(N);
    mu  = X(perm(1:K),:);
  case 'farthest',
    mu  = kInit(X,K, true);
  case 'k++',
    mu  = kInit(X,K, false);
  otherwise, error('Unknown initialization method');
  end;
else
  mu  = cInit;
  if (isempty(K)) K=size(cInit,1); end;
end;
Sig = zeros(D,D,K); for c=1:K, Sig(:,:,c) = eye(D); end;
alpha  = ones(1,K)./K;
R = zeros(N,K);

%%%%%%%%%%%%%%%%%
%% Optimization
%%%%%%%%%%%%%%%%%
iter = 1;
ll = inf;  
llOld = inf;
done = (iter>nIter);
C = log(2*pi)*D/2;
if (doPlot) llplot = zeros(1,nIter); end;

while (~done) 
   ll = 0;
   for c=1:K,
     V = X - repmat(mu(c,:),[N,1]);    %compute log prob of all data under model c
     R(:,c) = -.5 * sum((V * inv(Sig(:,:,c))) .* V,2) - .5* log(det(Sig(:,:,c))) + log(alpha(c)) - C;
   end;
   mx = max(R,[],2);            % avoid numerical issues by removing constant 1st
   R  = R - repmat(mx,[1,K]);   %  
   R  = exp(R);                 % then exponentiate and compute sum over components
   nm = sum(R,2);
   ll = sum( log(nm)+mx );      % update log-likelihood of the data
   R  = R ./ repmat(nm,[1,K]);  % normalize to give membership probabilities
   
   alpha = sum(R,1);                              % total weight for each component
   if (doPlot && D==2) fig(1); hold off; plotClassify2D([],X,from1ofK(R)); end;
   for c=1:K,
     mu(c,:)= (R(:,c)./alpha(c))'*X;                         % weighted mean estimate
     tmp   = X - repmat(mu(c,:),[N,1]);                  
     Sig(:,:,c)= tmp' * (tmp .* repmat(R(:,c)/alpha(c),[1,D])) + reg*eye(D);  % weighted covar estimate
     if (doPlot && D==2) fig(1); hold on; plotGauss2D(mu(c,:),Sig(:,:,c),'k-','linewidth',2); drawnow; end;
   end;
   alpha = alpha./N;
   if (doPlot) llplot(iter)=ll; fig(2); plot(1:iter,llplot(1:iter),'b-'); drawnow; end;
   %pause(.1);

   done = (iter >= nIter) || (abs(ll - llOld)<tol);  % stopping criteria
   llOld = ll;
   iter = iter+1;
end;
T.pi = alpha; T.mu = mu; T.Sig = Sig;
soft = R;

if (iter >= nIter) warning('emclust:iter','emclust:iter :: stopped after reaching maximum number of iterations'); end;

z = from1ofK(R);

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


