                          SVM DEMO CODE v1.1
               Andrea Vedaldi <vedaldi@robots.ox.ac.uk>

Start MATLAB and try one of the following commands:

  svmdemo_linear ;  % linear SVM demo
  svmdemo_rbf ;     % RBF SVM demo
  svmdemo_compare ; % compares different kernels and parameters

to run a couple of demos comparing the linear and RBF kernels with
different settings. Use GENDATA() and SVMDEMO() to run a specific
demo. E.g.

  [X, y] = gendata(15, 15) ;
  svmdemo(X, y, 'rbf', 1, .5)

To use the learning code SVM(), to fit data points X with labels Y you
need to pre-compute the  Gram matrix K. For instance:

  X = [-1 -1 1 ; -1 -1 0] ; % data vectors
  y = [1 1 -1] ; % labels
  K = X'*X ; % linear kernel
  C = 1 ;
  model = svm(K,y,C) ;

See the source of SVMDEMO() for details.






                      == LINEAR SVM TRAINING ==

Given a vector X(i), the SVM of parameters w, b is the function

  F(X(i) ; w,b) = X(i)' w + b.

The sign of F yields the label (either -1 or +1) predicted for point
X(i):

  y = f(X(i) ; w,b) = sign F(X(i) ; w,b)

1. LEARNING AND PRIMAL PROBLEM

Given the training data X(1), y(1), ... , X(n), y(n), we want to find
w, b so that most of the predicted labels correspond to the ground
truth labels (i.e. to fit the binary function y = f(X ; w,b) to the
data). The prediction error is measured by the average 01 risk:

  ERR_01(w,b ; X(1), ..., X(n))
    = 1/n sum_{i=1}^n err_01(y(i) F(X(i) ; w,b))

where err01(z) = (1 - sign z) / 2 is the 01 loss. Notice that
err01(y(i) F(X(i) ; w, b)) is 1 if F(X(i) ; w,b) != y(i) and zero otherwise,
so that ERR_01 is just the average number of incorrect predictions.

err_01(z) is a non-convex function. To simplify optimization, we
consider instead a convex upper bound given by the so called hinge
loss:

  err_01(z) <= err_hinge(z) = max{0, 1 - z}.

So we would like to find the parameters w,b that minimize the average
hinge error

  ERR_hinge(w,b ; X(1), ..., X(n))
     = 1/n sum_i err_hinge(y(i) F(X(i) ; w,b))

Let xi(i) be a scalar (slack variable) and consider the problem

    min_{xi(i)}  xi(i)
        s.t.  xi(i) >= 0,
              xi(i) >= 1 - y(i) F(X(i) ; w, b).

The solution is clearly xi(i) = err_hinge(y(i) ; F(X(i) ; w,b). Thus
minizing ERR_hinge(w,b) is equivalent to minimizing

    min_{w,b,xi(1),...,xi(n)} 1/n \sum_i xi(i),
        s.t. xi(i) >= 0,
             xi(i) >= 1 - y(i) F(X(i) ; w, b).

Let xi be the vector [xi(1) ... xi(n)'] of slack variables, 1 the
vector of all ones, and Y = diag(y). Then, recalling the definition of
F(X(i) ; w,b), we can rewrite this problem compactly as

    min_{w,b,xi} 1'xi,
        s.t.  xi >= 0,
              xi >= 1 - Y(X'w + 1 b)

Solving this problem yields an SVM F(X ; w*,b*) that fits the data as
well as possible. However, thee is no control on the "complexity" of
the resulting function F(X ; w*,b*). We can control this by imposing
that F(X ; w,b) is slowly varying. Since the derivative of F w.r.t. x
is exactly the parameter vector w, we can impose that the norm of w is
small (small derivative). This yields the SVM primal problem:

    min_{w,b,xi} .5 w'w + C 1'xi,
        s.t.  xi >= 0,
              xi >= 1 - Y(X'w + 1 b)

where C controls the trade-off between fitting the data and having a
simple (slowly varying) F.

2. DUAL PROBLEM

Introducing the Lagrange multiplier (dual vector) alpha, we can
formulate the equivalent problem

  min_{w,b,xi>=0} max_{alpha>=0} .5 w'w + C 1'xi - alpha' (xi - 1 + Y(X'w + 1b))

Switching min with max we obtain, for a given alpha >= 0, the problem

  min_{w,b,xi>=0} .5 w'w + C 1'xi - alpha' (xi - 1 + Y(X'w + 1b))   (*)

By collecting the terms in xi and b:

 min_{w,b,xi>=0} .5 w'w + (C 1 - alpha)' xi + alpha'(1- YX'w) - alpha' y b.

This problem has a trivial solution (-infinity) obtained by letting
components of xi or b go to infinity unless two conditions are
satisfied:

(1)  alpha' y = 0   and
(2)  C1 - alpha >= 0

If (1) and (2) are satisfied, then the minimum is bounded. Moreover if
the second condition is strictly satisfied for the i-th constraint,
i.e.  alpha(i) < C, then the optimal value for the i-th slack variable
xi(i) is 0 (because xi(i) multiplies a positive number C -
alpha(i)). If instead alpha(i) = C, then xi(i) is multiplied by zero
in the cost function and it does not influence its value. Similarly,
if (1) is satisfied b has no effect on the cost function. It remains
to determine w by solving

  min_w .5 w'w + alpha' (1 - YX'w).

The minimizer is found by deriving w.r.t to w and equating to zero,
which yields

  w = X Y alpha,      (**)

which corresponds to the objective

  -0.5 alpha' YX'XY alpha + alpha' 1

We need to maximize this w.r.t. alpha:

 max_{alpha} -0.5 alpha' YX'XY alpha + alpha' 1, s.t.
     0 <= alpha <= C,
     alpha' y = 0.

This is the dual problem. We denote alpha* it solution.

3. RECOVERING PRIMAL PARAMETRS

Given the soulution alpha* to the dual problem, we must recover the
primal parameters w* and b* (and optionally the slacks xi*). w* is
obtained from (**) as

  w* = X Y alpha*.

b* can be computed from the support vectors on the margin. Recall that
a support vectors is a point X(i) whose primal constraint is active,
i.e.

             xi*(i) = 1 - Y(i) (X(i)'w* + b)

These constraints can be identified by the condition alpha*(i) > 0. To
see why, notice given the optimal primal parameters w*, b* and xi*,
alpha* can be obtained as the minimizer of

  max_{alpha>=0} .5 w*'w* + C 1'xi* - alpha' (xi*-1+Y(X'w*+1b*))

Thus if the i-th component of xi*-1+Y(X'w*+1b*) is strictly larger
than zero (so that the i-th constraints is NOT active) then alpha*(i)
must be equal to zero (otherwise we could increase the cost function
by lowering alpha*(i)). Hence all constraints for which alpha*(i) > 0
must be active.

Along with the conditions we found before on alpha*(i) < C, we have
thus

    alpha*(i) > 0    ==>   xi*(i) = 1 - y(i) (X(i)'w* + b)  [active constraint]
    alpha*(i) < C    ==>   xi*(i) = 0                       [null slack] (***)

The points X(i) that satisfy both conditions, i.e. 0 < alpha*(i) < C,
satisfy

    0 = xi*(i) = 1 - y(i) (X(i)'w* + b*)

from which we can recover b* given w* = X Y alpha*:

    b* = 1 - y(i) X(i)'w*
       = y(i) - X(i)'w*

Notice that such points X(i) are support vectors and are on the
margin, in the sense that evaluating the SVM at X(i) yields either -1
or +1:

    0 < alpha*(i) < C  ==>  F(X(i) ; w*,b*) = X(i)'w* + b* = y(i)

4. SPECIAL CASES

If C is very small (severe underfitting), there may be no points on
the margin. In this case we must do more work to recover b*.  In this
case alpha*(i) is either 0 or C for all points. w* is obtained as
before as w* = X Y alpha*, but we must look at the primal to recover
b* (and optionally xi*):

  min_{b,xi>=0} .5 w*'w* + C 1'xi, s.t.
                 xi >= 1 - Y(X'w* + 1 b),

By exploiting the properties (***) we can substitute xi*(i) with
its optimal value as a function of b and get

  min_{b} const. - C sum_{i : alpha*(i) = C} Y(i) b
   s.t. 1 - Y(i)X(i)'w* <= Y(i)b  if alpha*(i) = 0 and
        1 - Y(i)X(i)'w* >= Y(i)b  if alpha*(i) = C.

The constraints can be further expandend in the conditions:

   b <=  1 - X(i)'w*        Y(i) > 0  and alpha*(i) = C
   b <= -1 + X(i)'w*        Y(i) < 0  and alpha*(i) = 0
   b >= -1 + X(i)'w*        Y(i) < 0  and alpha*(i) = C
   b >=  1 - X(i)'w*        Y(i) > 0  and alpha*(i) = 0

Depending on whetehr sum_{i : alpha*(i) = C} Y(i) is larger or smaller
than zero, b* is then found as either the minmum of such upper bounds,
or as the maximum of the lower bounds respectively.





               == DERIVATION FOR THE NON-LINEAR CASE ==

The non-linear case is essentially identical, except that X(i) should
be thought as the output of a feature map applied to the i-th data
point x(i), i.e. X(i) = psi(x(i)). The map psi() represents the given
kernel, in the sense that

  < psi(x(i)), psi(x(j)) > = X(i)'X(j) = kernel(x(i), x(j))

In the dual problem, nothing changes as X occurs only multiplied to
itself. Specifically, we call X'X = K the Gram matrix and we obtain
alpha* as the minimizer of

   min_{alpha} 0.5 alpha' YKY alpha - alpha' 1, s.t.
     0 <= alpha <= C,
     alpha' y = 0.

While w* = X Y alpha* cannot be written explicitly, we can still
compute b* as

 b* = y(i) - X(i)'w* = y(i) - X(i)'XY alpha* = y(i) - K(i,:) Y alpha*

Finally, to evaluate the SVM on a new test point x(test) we
still do not need to compute X(test) = psi(x(test)) explicitly,
but we can use

  F(x ; alpha*, b*) = X(i)'XYalpha* + b*
                    = kernel(x(i), X) Y alpha* + b*
