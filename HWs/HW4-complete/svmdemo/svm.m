function model = svm(K,y,C)
% SVM  Learns non-linear SVM model
%   MODEL = SVM(K, Y, C) where K is the Gram matrix, Y the label
%   vector and C the SMV constant. The function optimizes the dual
%   problem
%
%     min_{alpha} .5 * alpha' diag(Y) K diag(Y) alpha - 1' * alpha,
%         s.t.   0 <= alpha <= C,
%                y' alpha = 0.
%
%   where 1 is a vector of all ones.
%
%   Recall that, given data points X(1), ..., X(N), their Gram matrix
%   is the matrix of all inner products (kernel evaluations):
%
%     K(i,j) = kernel(X(i), X(j)).
%
%   The vecotr Y is a N dimensional vector of +1 and -1 with the data
%   labels.
%
%   The function returns a structure MODEL with the following fields:
%
%   MODEL.ALPHA::
%     Minimizer of the dual problem.
%
%   MODEL.SVIND::
%     Indexes of the support vectors (points I for which ALPHA(I) >
%     0).
%
%   MODEL.BNDIND::
%     Indexes of the vectors on the decision boundary (points for
%     which 0 < ALPHA(I) < C).
%
%   MODEL.B::
%     SVM offset.
%
%   Given a new test point X(test) the SVM can be evaluated as:
%
%      sum_{i in SVIND} ALPHA(i) Y(i) kernel(X(i), X(test)) + B
%
%   Author:: Andrea Vedaldi (vedaldi@robots.ox.ac.uk)

n = numel(y) ;
Y = diag(y) ;
tol = 1e-4 ;
alpha = quadprog(Y*K*Y, - ones(n,1), ...
                 [], [], ...
                 y, 0, ...
                 zeros(n,1), C * ones(n,1), ...
                 [], optimset('display','off','largescale', 'off')) ;

model.bndind = find(alpha > tol * C & alpha < (1 - tol) * C) ;
model.svind = find(alpha > tol * C) ;
model.alpha = alpha ;
model.alphay = Y * alpha ;

if ~ isempty(model.bndind)
  % This worsk 99% of the times
  model.b = mean(y(model.bndind) - model.alphay' * K(:,model.bndind)) ;

else
  % Special cases to deal with the case in which C is very small
  % and there are no support vectors on the margin

  r = 1 - model.alphay' * K * Y ;
  act = ismember(1:n, model.svind) ;
  pos = y > 0 ;

  maxb = min([+r(pos & act),  -r(~pos & ~act)]) ;
  minb = max([-r(~pos & act), +r(pos & ~act)]) ;
  if mean(y(act)) <= -tol
    model.b = maxb ;
  elseif mean(y(act)) > tol
    model.b = minb;
  else
    % any b in the interval is equivalent
    model.b = mean([minb maxb]) ;
  end
end
