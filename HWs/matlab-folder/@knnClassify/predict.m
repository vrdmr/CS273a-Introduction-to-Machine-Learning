function Yte = predict(obj,Xte)
% Yhat = predict(knn, Xtest) : make a nearest-neighbor prediction on test data

  [Ntr,Mtr] = size(obj.Xtrain);          % get size of training, test data
  [Nte,Mte] = size(Xte);
  Yte = repmat(obj.Ytrain(1), [Nte,1]);  % make Ytest the same data type as Ytrain
  K = min(obj.K, Ntr);                   % can't have more than Ntrain neighbors
  for i=1:Nte,                           % For each test example:
    dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);  % compute sum of squared differences
    %dist = sum( (obj.Xtrain - repmat(Xte(i,:),[Ntr,1]) ).^2 , 2);  % compute sum of squared differences
    [dst,idx] = sort(dist);              % find nearest neighbors over Xtrain
    dst=dst(1:K); idx=idx(1:K);          % keep nearest K data points
    wts = exp(-obj.alpha*dst);
    cMax=1; NcMax=0;                     % then find the majority class within that set of neighbors
    for c=1:length(obj.classes),
      Nc = sum(wts(obj.Ytrain(idx)==obj.classes(c)));  % add up weights of instances of that class we have
      if (Nc>NcMax), cMax=c; NcMax=Nc; end;        % save the largest count and its class id
    end;
    Yte(i)=obj.classes(cMax);            % save results
  end;

