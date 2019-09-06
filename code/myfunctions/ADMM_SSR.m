function [x_est,arrSER]=ADMM_SSR(y,A,arrR,matQ,invMat,lambda,rho,prox,nIteration,tol,x_true)
% ADMM_SSR sum of sparse regularizer (SSR) optimization based on alternating direction method of multipliers (ADMM)
%
% Input 
%   y: measurement vector
%   A: measurement matrix
%   arrR: array for candidate of unknown variables
%   matQ: coefficients in SSR optimization
%   invMat: inversion matrix in the algorithm 
%   lambda: parameter in SSR optimization
%   rho: parameter of ADMM
%   prox: proximity operator of sparse regularizer
%   nIteration: maximum number of iteration
%   tol: torelance
%   x_true: true unknown vector (only used in the evalution of SER)
% Output
%   x_est: estimate of unknown vector
%   arrSER: array of symbol error rate (SER)
%

  [M,N]=size(A);
  L=length(arrR);

  x_MF=A.'*y;
  Phi=kron(ones(L,1),eye(N));
  thre=reshape(matQ,N*L,1)/rho;
  orig=kron(arrR.',ones(N,1));
  s=zeros(N,1);
  z=zeros(L*N,1);
  w=zeros(L*N,1);
  arrSER=zeros(1,nIteration);
  arrSER(1)=nnz(quantize(zeros(N,1),arrR)-x_true)/N;
  for k=2:nIteration
    s_prev=s;
    s=invMat*(rho*Phi.'*(z-w)+lambda*x_MF);
    u=Phi*s+w;
    z=orig+prox(u-orig,thre);
    w=w+Phi*s-z;
    arrSER(k)=nnz(quantize(s,arrR)-x_true)/N;
    if norm(s-s_prev,2)^(2)/N<tol
      break;
    end
  end
  x_est=s;
  
end
