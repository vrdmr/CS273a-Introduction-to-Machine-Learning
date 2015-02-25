% SVMDEMO_RBF

Np = 15 ;
Nn = 15 ;
C  = 1 ;
gamma = .25 ;

[X, y] = gendata(Np, Nn) ;

cla ;
model = svmdemo(X, y, 'rbf', C, gamma) ;
