function obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps)
% obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes (binary, +1 or -1)
%     stepsize = step size for gradient descent ("learning rate")
%     tolerance = tolerance for stopping criterion
%     maxSteps = maximum number of steps before stopping
%
if (nargin < 6) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 5) tolerance = 1e-4; end;  % stopping tolerance
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

plotFlag = 1;                    % with plotting

[n,d] = size(Xtrain);            % d = dimension of data; n = number of training data
Xtrain1= [ones(n,1), Xtrain];    % make a version of training data with the constant feature

if (~all(Ytrain==+1 | Ytrain==-1)) error('Y values must be +/- 1'); end;  % check correct binary labeling
obj.classes = [-1 ; +1];

obj.wts = randn(1,d+1);          % initialize weights randomly

% Outer loop of stochastic gradient descent:
iter=1;                          % iteration #
done=0;                          % end of loop flag
err=zeros(1,maxSteps);           % misclassification rate values
while (~done) 
  % Step size evolution
  stepi = stepsize;              % perceptron method: usually fixed step size

  % Stochastic gradient update (one pass)
  for i=1:n,  % for each data example,
    resp = Xtrain1(i,:)*obj.wts';                       % compute linear response
    yhati = sign(resp);                                 % and prediction for Xi
    grad = (yhati - Ytrain(i))*Xtrain1(i,:);            % Gradient-like perceptron update rule
    obj.wts = obj.wts - stepi * grad;                   % Take a step down the gradient
  end;

  % Compute current error values
  err(iter)  = mean( (Ytrain~=sign(Xtrain1*obj.wts')) ); % misclassification rate

  % Make plots, if desired
  if (plotFlag),
  figure(1); semilogx(1:iter, err(1:iter),'g-');
  figure(2); switch d,                              % Plots to help with visualization
      case 1, plot1DLinear(obj,Xtrain,Ytrain);      %  for 1D data we can display the data and the function
      case 2, plot2DLinear(obj,Xtrain,Ytrain);      %  for 2D data, just the data and decision boundary
      otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
    end; 
  drawnow;
  end;

  done = (iter >= maxSteps || err(iter)==0);        % stop when no errors or out of time
  iter = iter + 1;
obj.wts,
end;

%figure(1); set(gca,'fontsize',20); print -depsc2 perceptErr.eps
figure(2); set(gca,'fontsize',20); print -depsc2 perceptClass.eps



