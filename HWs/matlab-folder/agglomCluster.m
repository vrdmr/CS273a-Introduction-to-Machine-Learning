function [z join] = agglomCluster(X,nClust,method,join)
% [z, join] = hcluster(X,K,method, [,join]) : perform heirarchical agglomerative clustering
%   z = assignment of data into K clusters
%   method = {'min','max,'means','average'}  : cluster distance / linkage type (single,complete,mean,avg)
%   join = sequence of joining operations performed by hcluster (pass to avoid re-clustering for new K)

[m,n] = size(X);         % get data size
D = zeros(1,m*(m-1)/2-m);  % store pairwise distances between clusters
z = (1:m)';              % assignments of data
num= ones(1,m);          % # of data in each cluster
mu = X;                  % centroid of each cluster

meth = 0;
Methods = {'min','max','means','average'};
if (nargin<3) method = 'means'; end;
for i=1:length(Methods) if (strcmp(Methods{i},lower(method))) meth=i; end; end;
if (meth==0) error(['hcluster: linkage method ', method, ' unknown; should be {min,max,means,average}']); end;


if (nargin < 4) %%%%%%%%%%%%%%%% if join not precomputed: %%

join = zeros(m-1,3);                     % keep track of join sequence

dist = @(a,b) sum( (a-b).^2 );           % use standard Euclidean distance

for i=1:m,                               %  and compute initial distances
  for j=i+1:m, 
    D(idx(i,j,m)) = dist(X(i,:),X(j,:));
  end; 
end;
open = true(1,m);                        % store list of clusters still in consideration
[val, k] = min(D);                       % find first join (closest cluster pair)

for c=1:m-1,
  [i j] = ij(k,m);
  join(c,:) = [i j val];
  %fprintf('joining %d & %d\n',i,j);
  muNew = (num(i)*mu(i,:)+num(j)*mu(j,:))/(num(i)+num(j));  % centroid of new cluster
  % compute new distances to cluster i
  for jj=find(open),
    if (jj==i || jj==j) continue; end; 
    switch (meth)
    case 1, D(idx(i,jj,m)) = min( D(idx(i,jj,m)), D(idx(j,jj,m)) );  % single linkage (min dist)
    case 2, D(idx(i,jj,m)) = max( D(idx(i,jj,m)), D(idx(j,jj,m)) );  % complete linkage (max dist)
    case 3, D(idx(i,jj,m)) = dist( muNew , mu(jj,:));                % mean linkage (dist between centroids)
    case 4, D(idx(i,jj,m)) = (num(i)*D(idx(i,jj,m))+num(j)*D(idx(j,jj,m)))/(num(i)+num(j));  % average linkage
    end;
  end;
  open(j) = 0;                         % close cluster j (fold into i)
  num(i) = num(i)+num(j);              % update total membership in cluster i to include j
  mu(i,:)= muNew;                      % update centroid list

  for ii=1:m, if (ii~=j) D(idx(ii,j,m))=inf; end; end;   % remove cluster j from consideration as min
  [val, k] = min(D);                   % and find next smallest pair
end;

end; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% now compute cluster assignments give sequence of joins
z = (1:m)';
for c=1:m-nClust,
  z( z==join(c,2) ) = join(c,1); 
end;
uniq = unique(z); 
for c=1:length(uniq), z(z==uniq(c))=c; end;


% compact index conversion:  (i,j) pair to compressed linear index
function k = idx(i,j,m)
  if (i>j) a=i; i=j; j=a; end;   % only stores 1/2 of distance matrix
  if (i==j) error('No access to diagonal values'); end;  % no self-distances
  k = m*(i-1) - i*(i-1)/2 + j - i;   % return compressed index
 
% compact index conversion:  compressed linear index to (i,j) pair
function [i,j] = ij(k,m)
  i=1; 
  while (k>m-i) k=k-(m-i); i=i+1; end;
  j=k+i;


%function tree=seq2tree(join)
%  for c=1:length(join)
%    i=join(c,1); j=join(c,2);
%    idx(i) = 
%  end;


