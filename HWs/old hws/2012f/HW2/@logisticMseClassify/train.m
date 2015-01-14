function obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps, init)
% obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps, init)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes (-1 or +1)
%     stepsize = step size for gradient descent (decreases as 1/iter)
%     tolerance = tolerance for stopping criterion
%     maxSteps = maximum number of steps before stopping
%     init = 'keep', 'zeros', 'randn', 'linreg' 
%
if (nargin < 7) init = 'zeros';   end;  % initialization method
if (nargin < 6) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 5) tolerance = 1e-4; end;  % stopping tolerance
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

plotFlag = 1;                    % with plotting

[n,d] = size(Xtrain);            % d = dimension of data; n = number of training data
Xtrain1= [ones(n,1), Xtrain];    % make a version of training data with the constant feature

if (~all(Y==+1 | Y==-1)) error('Y values must be +/- 1'); end;  % check correct binary labeling
classes = [-1 ; +1];

obj.wts = randn(1,d+1);          % initialize weights randomly

% Initialize weights if desired:
switch lower(init),
  case 'keep',    if (length(obj.wts)~=d+1) obj.wts=zeros(1,d+1); end; % try to keep current value
  case 'zeros',   obj.wts = zeros(1,d+1);                              % init to all-zero
  case 'randn',   obj.wts = randn(1,d+1);                              % init at random
  case 'linreg', idx=randperm(n); idx=idx(1:min(max(4*d,100),n));      % initialize w/ a small linear regression
          obj.wts = Ytrain(idx)'*Xtrain1(idx,:)*inv(Xtrain1(idx,:)'*Xtrain1(idx,:)+.1*eye(d+1));
end;
wtsOld = 0*obj.wts+inf;                                                % save for convergence checks

% Outer loop of stochastic gradient descent:
iter=1;                                               % iteration #
done=0;                                               % end of loop flag
surr=zeros(1,maxSteps);                               % surrogate loss values
err=zeros(1,maxSteps);                                % misclassification rate values
while (~done) 
  % Step size evolution
  stepi = stepsize / iter;                            % classic 1/t decrease

  % Stochastic gradient update (one pass)
  for i=1:n,  % for each data example,
    resp = Xtrain1(i,:)*obj.wts';                     % compute linear response
    yhati = sign(resp);                               % and prediction for Xi
   
    % Compute gradient of loss function 
    grad = ...     

    % Take a step down the gradient:
    obj.wts = obj.wts - stepi * grad; 
  end;

  % Compute current error values
  err(iter)  = ...                                       % misclassification rate\
  surr(iter) = ...                                       % surrogate loss = logistic MSE cost

  % Make plots, if desired
  if (plotFlag), 
    figure(1); plot(1:iter, surr(1:iter),'b-',1:iter,err(1:iter),'g-');
    figure(2); switch d,             % Plots to help with visualization
      case 1, plot1DLinear(obj,Xtrain,Ytrain);      %  for 1D data we can display the data and the function
      case 2, plot2DLinear(obj,Xtrain,Ytrain);      %  for 2D data, just the data and decision boundary
      otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
    end; 
  drawnow;
  end;

  % Various stopping conditions:
  %done = (iter >= maxSteps); 
  done = (iter>1) && ( (abs(surr(iter)-surr(iter-1))<tolerance) || (iter >= maxSteps) );
  %done = (abs(grad*grad')<tolerance) || (iter >= maxSteps);
  %done = (abs(wtsOld-obj.wts)<tolerance) || (iter >= maxSteps);
  iter = iter + 1;
  wtsOld = obj.wts;
end;




