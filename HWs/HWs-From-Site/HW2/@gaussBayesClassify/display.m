function display(obj)
% display(gbc) : output information about the classifier
  fprintf('Gaussian classifier, %d classes:\n',length(obj.classes));
  disp(obj.classes');
  fprintf('Means:\n'); disp(obj.means);
  fprintf('Covariances:\n'); disp(obj.covars);
