function p = predictSoft(obj,Xte)
% Prob = predictSoft(obj,Xtest) : make "soft" predictions on test data with the classifier
  [m n] = size(Xte);
  C = length(obj.classes);
  p = zeros(m,C);
  for c=1:C,                                % compute probabilities for each class by Bayes rule
    p(:,c) = obj.probs(c) * evalGaussian( Xte, obj.means{c},obj.covars{c} );   % p(c) * p(x|c) 
  end;                                     
  p = p ./ repmat(sum(p,2),[1,C]);          % normalize each row (data point)
end


function p = evalGaussian( X , gMean, gCov )
  d = size(X,2); n = size(X,1);               % get dimension and # of data
  p = zeros(n,1);                             % store evaluated probabilities for each datum
  constant = 1/(2*pi)^(d/2) / det(gCov)^(.5); % normalization constant for Gaussian
  invCov = inv(gCov);                         % need inverse covariance
  %for i=1:size(X,1),                          % compute probability of Gaussian at each point
  %  p(i) = exp(-.5 * (X(i,:)-gMean)*invCov*(X(i,:)-gMean)' ) * constant; 
  %end;
  R = X - repmat(gMean,[n,1]);                % compute probability of Gaussian at each point
  p = exp(-.5 *  sum( (R*invCov).*R ,2)) *constant;  % (vectorized)
end
