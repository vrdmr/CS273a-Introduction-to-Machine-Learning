function c = covars(obj,c)
% c = covars(gbc [, c]) : access or set the covariances of the classifier

if (nargin > 1) 
  obj.covars = c; 
end;
c = obj.covars;

