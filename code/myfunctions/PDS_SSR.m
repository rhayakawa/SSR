function [x_est,arrSER]=SSRM_PDS(y,A,arrR,matQ,lambda,rho1,rho2,prox,nIteration,tol,x_true)
% PDS_SSR sum of sparse regularizer (SSR) optimization based on primal-dual splitting (PDS)
%
% Input 
%   y: measurement vector
%   A: measurement matrix
%   arrR: array for candidate of unknown variables
%   matQ: coefficients in SSR optimization
%   lambda: parameter in SSR optimization
%   rho1, rho2: parameters of primal dual splitting
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

  Phi=kron(ones(L,1),eye(N));
  thre=reshape(matQ,N*L,1)/rho2;
  orig=kron(arrR.',ones(N,1));
  s=zeros(N,1);
  w=zeros(L*N,1);
  arrSER=zeros(1,nIteration);
  arrSER(1)=nnz(quantize(zeros(N,1),arrR)-x_true)/N;
  for k=2:nIteration
    s_prev=s;
    s=s_prev-rho1*(lambda*A.'*(A*s_prev-y)+Phi.'*w);
    z=w+rho2*Phi*(2*s-s_prev);
    w=z-rho2*(orig+prox(z/rho2-orig,thre));
    arrSER(k)=nnz(quantize(s,arrR)-x_true)/N;
    if norm(s-s_prev,2)^(2)/N<tol
      break;
    end
  end
  x_est=s;
  
end
