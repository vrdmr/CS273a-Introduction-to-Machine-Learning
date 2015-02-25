function m = means(obj,m)
% m = means(obj [, m]) : access or set the classifiers means

if (nargin > 1)
  obj.means = m;
end;
m = obj.means;
