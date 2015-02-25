function [X Z] = dataGMM(N,C,d)

if (nargin<3) d=2; end;
%if (nargin<2) end;

%pi = random('Gamma',1,C);   % use stick-breaking for DPMM?
for c=1:C, pi(c)=gamrand(10,.5); end;
pi = pi/sum(pi);
cpi = cumsum(pi);

Rho = rand(d,d); Rho = Rho+Rho'; Rho = Rho + d*eye(d);
Rho = sqrtm(Rho);

mu = randn(c,d)*Rho;

for i=1:C,
  tmp = rand(d,d); tmp = tmp+tmp'; tmp = .05*(tmp + d*eye(d));
  CCov{i} = sqrtm(tmp);
end;

p = rand(N,1); Z = ones(N,1);
for c=1:C-1, Z(p>cpi(c))=c+1; end;
X = mu(Z,:);
for c=1:C, X(Z==c,:) = X(Z==c,:) + randn(sum(Z==c),d)*CCov{c}; end;


function x=gamrand(alpha,lambda)
% Gamma(alpha,lambda) generator using Marsaglia and Tsang method
% Algorithm 4.33
if alpha>1
    d=alpha-1/3; c=1/sqrt(9*d); flag=1;
    while flag
        Z=randn;
        if Z>-1/c
            V=(1+c*Z)^3; U=rand;
            flag=log(U)>(0.5*Z^2+d-d*V+d*log(V));
        end
    end
    x=d*V/lambda;
else
    x=gamrand(alpha+1,lambda);
    x=x*rand^(1/alpha);
end
