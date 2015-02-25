function display(obj)
  fprintf('KNN Classifier, %d classes, K=%d',length(obj.classes), obj.K);
  if (obj.alpha ~= 0) fprintf(', weighted (alpha=%f)',obj.alpha); end;
  fprintf('\n');

