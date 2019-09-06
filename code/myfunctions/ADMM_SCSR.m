function [x_est,arrSER]=SCSR_ADMM(y,A,arrC,matQ,invMat,lambda,rho,prox,nIteration,tol,x_true)
% SCSR_ADMM sum of complex sparse regularizers (SCSR) optimization based on alternating direction method of multipliers (ADMM)
%           for discrete-valued sparse vector 
%
% Input 
%   y: measurement vector
%   A: measurement matrix
%   arrC:  array for candidate of unknown variables
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
  L=length(arrC);

  x_MF=A'*y;
  Phi=kron(ones(L,1),eye(N));
  thre=reshape(matQ,N*L,1)/(2*rho);
  orig=kron(arrC.',ones(N,1));
  s=zeros(N,1);
  z=zeros(L*N,1);
  w=zeros(L*N,1);
  arrSER=zeros(1,nIteration);
  arrSER(1)=nnz(quantize_comp(zeros(N,1),arrC)-x_true)/N;
  for k=2:nIteration
    s_prev=s;
    s=invMat*(rho*Phi.'*(z-w)+lambda*x_MF);
    u=Phi*s+w;
    prox_real=prox(real(u-orig),thre);
    prox_imag=prox(imag(u-orig),thre);
    prox_modu=prox(abs(u-orig),thre).*(u-orig)./abs(u-orig);
    z=orig+(prox_real+1j*prox_imag).*(abs(orig)>0)+prox_modu.*(abs(orig)==0);
    w=w+Phi*s-z;
    arrSER(k)=nnz(quantize_comp(s,arrC)-x_true)/N;
    if norm(s-s_prev,2)^(2)/N<tol
      break;
    end
  end
  x_est=s;
  
end
