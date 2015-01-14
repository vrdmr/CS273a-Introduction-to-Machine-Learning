function display(obj)
  fprintf('KNN Regression, K=%d', obj.K);
  if (obj.alpha ~= 0) fprintf(', weighted (alpha=%f)',obj.alpha); end;
  fprintf('\n');

