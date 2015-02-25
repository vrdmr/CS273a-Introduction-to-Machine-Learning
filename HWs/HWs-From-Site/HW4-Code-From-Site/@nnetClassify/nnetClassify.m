function obj = nnetClassify(Xtr,Ytr, sizes, init, varargin)
% nnetClassify(Xtr,Ytr,sizes,init,...) : construct a neural network classifier
%   can take no arguments, or for full args see nnetClassify/initWeights and train
%   sizes = [Nin, Nh1,...,Nout] where Nout is the # of outputs (usually, # classes)
%   member weights are {W1,...,WL-1}, where W1 is Nh1 x Nin, etc.

% TODO: check # classes & output dimension; conversions to 1-of-K representations?
%   (better not to convert?  => ok for structured output models) (how to make consistent with other outputs?)

  if (nargin < 4) init = 'zeros'; end;

  obj.classes = [];
  obj.wts=cell(1,length(sizes)-1);
  obj.activation=[]; obj.Sig=[]; obj.dSig=[]; obj.Sig0=[]; obj.dSig0=[];

  obj=class(obj,'nnetClassify');

  obj=setActivation(obj,'logistic');        % default to logistic activations
  obj=initWeights(obj,sizes,init,Xtr,Ytr);  % default initialization
  if (nargin > 0 && ~isempty(Xtr)) 
    obj=train(obj,Xtr,Ytr, varargin{:});    % train if data available
  end;

