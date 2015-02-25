function display(obj)
% display(learner) : display the coefficients of the linear regression object
  fprintf('Linear Regression Object; %d features\n',length(obj.theta));
  disp(obj.theta);

