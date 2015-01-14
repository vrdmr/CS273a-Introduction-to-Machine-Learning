function Prob = predictSoft(obj,Xte)
% Prob = predictSoft(knn, Xtest) : make a "soft" nearest-neighbor prediction on test data

  [Ntr,Mtr] = size(obj.Xtrain);          % get size of training, test data
  [Nte,Mte] = size(Xte);
  nClasses  = length(obj.classes);
  Prob = zeros(Nte,nClasses);            % allocate memory for class probabilities
  K = min(obj.K, Ntr);                   % can't have more than Ntrain neighbors
  for i=1:Nte,                           % For each test example:
    %dist=zeros(1,Ntr);
    %for j=1:Ntr, dist(j)=sum( (obj.Xtrain(j,:)-Xte(i,:)).^2 ); end;
    %dist = sum( (obj.Xtrain - repmat(Xte(i,:),[Ntr,1]) ).^2 , 2);  % compute sum of squared differences
    dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);  % compute sum of squared differences
    [dst,idx] = sort(dist);              % find nearest neighbors over Xtrain
    dst=dst(1:K); idx=idx(1:K);          % keep nearest K data points
    wts = exp(-obj.alpha*dst);
    count = zeros(1,nClasses);
    for c=1:length(obj.classes),      
      count(c) = sum(wts(obj.Ytrain(idx)==obj.classes(c)));  % total weight of instances of that class
    end;
    Prob(i,:) = count / sum(count);      % save (soft) results
  end;

