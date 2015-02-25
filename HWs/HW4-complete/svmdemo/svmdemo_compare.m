% Runs several demos in sequence

Np = 15 ;
Nn = 15 ;
randn('state', 1) ;

[X, y] = gendata(Np, Nn) ;

% linear kernel
figure(1) ; clf ;
Cr = [.005 .01 1] ;

t = 0 ;
for C = Cr
  t = t + 1 ;
  subplot(3,1,t) ;
  randn('state', 1) ;
  svmdemo(X, y, 'linear', C) ;
end

% RBF kernel
figure(2) ; clf ;
Cr = [.125 1 5] ;
gammar = [.1 .5 1] ;

t = 0 ;
for C = Cr
  for gamma = gammar
    t = t + 1 ;
    subplot(3,3,t) ;
    randn('state', 1) ;
    svmdemo(X, y, 'rbf', C, gamma) ;
  end
end

