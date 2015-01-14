function Yte = predict(obj,Xte)
% Yhat = predict(knn, Xtest) : make a nearest-neighbor prediction on test data

  [Ntr,Mtr] = size(obj.Xtrain);         % get size of training, test data
  [Nte,Mte] = size(Xte);
  if (Mtr ~= Mte) error('Training and prediction data have different numbers of features!'); end;
  Yte = repmat(obj.Ytrain(1), [Nte,1]);	% make Ytest the same data type as Ytrain
  K = min(obj.K, Ntr);		        % can't have more than Ntrain neighbors
  for i=1:Nte,
    dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);	% compute sum of squared differences
    %dist = sum( (obj.Xtrain - repmat(Xte(i,:),[Ntr,1]) ).^2 , 2);	% compute sum of squared differences
    [dst,idx] = sort(dist);             % find nearest neighbors over Xtrain
    dst=dst(1:K); idx=idx(1:K);         % keep nearest K data points
    wts = exp(-obj.alpha*dst);

    %Yte(i)=mean(obj.Ytrain(idx(1:K)));     % straight average
    Yte(i)=wts'*obj.Ytrain(idx)./sum(wts);  % weighted average
  end;

