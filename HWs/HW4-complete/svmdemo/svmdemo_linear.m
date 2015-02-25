% SVMDEMO_LINEAR

Np = 15 ;
Nn = 15 ;
C  = .1 ;

[X, y] = gendata(Np, Nn) ;

cla ;
model = svmdemo(X, y, 'linear', C) ;
