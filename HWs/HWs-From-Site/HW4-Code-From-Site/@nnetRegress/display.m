function display(obj)
% display function for object
  fprintf('Multi-layer perceptron (neural network) regression object; ');
  layers = getLayers(obj);
  fprintf('[ '); fprintf('%d ',layers); fprintf(']\n');


