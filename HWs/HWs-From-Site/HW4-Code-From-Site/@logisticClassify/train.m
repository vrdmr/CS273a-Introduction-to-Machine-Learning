function obj = train(obj, X, Y, varargin)
% obj = train(obj, Xtrain, Ytrain [, option,val, ...])  : train logistic classifier
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes 
%     'stepsize', val  => step size for gradient descent [default 1]
%     'stopTol',  val  => tolerance for stopping criterion [0.0]
%     'stopIter', val  => maximum number of iterations through data before stopping [1000]
%     'reg', val       => L2 regularization value [0.0]
%     'init', method   => 0: init to all zeros;  1: init to random weights;  
% Output:
%   obj.wts = [C-1 x d+1] vector of weights; wts(1) + wts(2)*X(:,1) + wts(3)*X(:,2) + ...

% Unused options
%     'plot', val      => 0=no plots, 1=plot 1D prediction against X(1,:); 2=plot 2D decision boundary


  [n,d] = size(X);            % d = dimension of data; n = number of training data

  % default options:
  plotFlag = true; 
  init     = []; 
  stopIter = 1000;
  stopTol  = -1;
  reg      = 0.0;
  stepsize = 1;

  i=1;                                       % parse through various options
  while (i<=length(varargin)),
    switch(lower(varargin{i}))
    case 'plot',      plotFlag = varargin{i+1}; i=i+1;   % plots on (true/false)
    case 'init',      init     = varargin{i+1}; i=i+1;   % init method
    case 'stopiter',  stopIter = varargin{i+1}; i=i+1;   % max # of iterations
    case 'stoptol',   stopTol  = varargin{i+1}; i=i+1;   % stopping tolerance on surrogate loss
    case 'reg',       reg      = varargin{i+1}; i=i+1;   % L2 regularization
    case 'stepsize',  stepsize = varargin{i+1}; i=i+1;   % initial stepsize
    end;
    i=i+1;
  end;

stepsize,

  X1    = [ones(n,1), X];     % make a version of training data with the constant feature

  Yin = Y;                              % save original Y for use later
  [Y, obj.classes] = toIndex(Yin);      % & convert class values to index (1..C)
  C = length(obj.classes);

  if (~isempty(init) || isempty(obj.wts))   % initialize weights and check for correct size
    obj.wts = initWeights(obj,X,Y,init);
  end;
  if (any( size(obj.wts) ~= [C-1 d+1]) ) error('Weights are not sized correctly for these data'); end;
  wtsold = 0*obj.wts+inf;

% Training loop (SGD):
iter=1; Jsur=zeros(1,stopIter); J01=zeros(1,stopIter); done=0; 
while (~done) 
  step = stepsize/iter;               % update step-size and evaluate current loss values
  Jsur(iter) = nll(obj,X,Yin) + reg*sum(obj.wts(:).^2);
  J01(iter) = err(obj,X,Yin);

  if (plotFlag), switch d,            % Plots to help with visualization
    case 1, fig(2); plot1DLogistic(obj,X,Y);  %  for 1D data we can display the data and the function
    case 2, fig(2); plot2DLogistic(obj,X,Y);  %  for 2D data, just the data and decision boundary
    otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
  end; end;
  fig(1); semilogx(1:iter, Jsur(1:iter),'b-',1:iter,J01(1:iter),'g-'); drawnow;

  for i=1:n,
    % Compute linear responses and activation
    z = [0 X1(i,:)*obj.wts'];         % compute linear response (1st class = 0)
    s = exp( z - max(z) );            % exponentiate responses (remove max for stability)

    % Compute gradient:
    grad = s' ./ sum(s) * X1(i,:);    % derivative of denominator, 1/sum(exp(f))
    grad(Y(i),:) = grad(Y(i),:) - X1(i,:);   % derivative of numerator, only for true class
    grad = grad(2:end,:);             % drop 1st class (parameters = 0)
    grad = grad + reg*obj.wts;        % include regularization derivative

    % Binary case:
    %grad = -(Y(i)==2)*(1-s(2))*X1(i,:) + (Y(i)==1)*s(2)*X1(i,:) + reg*obj.wts; 

    obj.wts = obj.wts - step * grad;      % take a step down the gradient
  end;

  done = false;
  if (iter > stopIter) done = true; end;
  if ((iter > 1) && abs(Jsur(iter)-Jsur(iter-1))<stopTol) done=true; end;
  % could check other stop conditions; hard to check gradient magnitude for SGD

  wtsold = obj.wts;
  iter = iter + 1;
end;


