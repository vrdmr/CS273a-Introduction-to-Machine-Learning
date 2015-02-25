function obj = gaussBayesClassify(Xtr,Ytr, varargin)
% gaussBayesClassify(X,Y,...) : train a Bayes classifier with Gaussian class-conditional distributions
% takes no arguments, or see gaussBayesClassify/train for training options
  obj.means={}; obj.covars={}; obj.probs=[]; obj.classes=[];
  obj=class(obj,'gaussBayesClassify');
  if (nargin > 0) 
    obj = train(obj,Xtr,Ytr, varargin{:});
  end;
end

