% display function, print out coefficients
function display(obj)
  fprintf('Logistic Regression Object; %d classes, %d features\n',length(obj.classes),size(obj.wts,2)-1);
  disp(obj.wts);

