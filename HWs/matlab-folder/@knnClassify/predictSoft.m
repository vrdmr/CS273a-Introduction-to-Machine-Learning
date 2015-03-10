function Prob = predictSoft(obj,Xte)
% Prob = predictSoft(knn, Xtest) : make a "soft" nearest-neighbor prediction on test data

  [Ntr,Mtr] = size(obj.Xtrain);          % get size of training, test data
  [Nte,Mte] = size(Xte);
  Prob = zeros(Nte,length(obj.classes)); % allocate memory for class probabilities
  K = min(obj.K, Ntr);                   % can't have more than Ntrain neighbors
  for i=1:Nte,                           % For each test example:
    dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);  % compute sum of squared differences
    %dist = sum( (obj.Xtrain - repmat(Xte(i,:),[Ntr,1]) ).^2 , 2);  % compute sum of squared differences
    [dst,idx] = sort(dist);              % find nearest neighbors over Xtrain
    dst=dst(1:K); idx=idx(1:K);          % keep nearest K data points
    wts = exp(-obj.alpha*dst);
    for c=1:size(Prob,2),                % then count up the weight associated with each class
      Nc = sum(wts(obj.Ytrain(idx)==obj.classes(c)));  % add up weights of instances of that class we have
      Prob(i,c) = Nc;                              % save the weight as our soft output
    end;
    Prob(i,:) = Prob(i,:)./sum(Prob(i,:));
  end;

