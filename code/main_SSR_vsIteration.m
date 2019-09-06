%%--------------------------------------------------------------------------------------------
% Binary vector reconstruction via sum of sparse regularizer (SSR) optimization
%
% Author:
%   Ryo Hayakawa
% Article:
%   Ryo Hayakawa and Kazunori Hayashi,
%   "Discrete-Valued Vector Reconstruction by Optimization with Sum of Sparse Regularizers,"
%   in Proc. EUSIPCO, Sept. 2019. 
%%--------------------------------------------------------------------------------------------

clear;
addpath("myfunctions");

%% problem setting
N=200;
Delta=0.8;
M=round(N*Delta);
SNR=15;
nIteration=500;
nSample=100;

% distribution of unknown vector
arrP=[1/2 1/2];
arrR=[-1 1];
arrQ_L1=[1/2 1/2];
arrQ_L0=[1/2 1/2];
arrQ_L1_2=[1/2 1/2];
arrQ_L2_3=[1/2 1/2];

% cumulative distribution
L=length(arrR);
matOne=ones(L,L);
arrCDF=arrP*triu(matOne);

% parameters of the proposed algorithm
lambda=0.05;
matQ_L1=ones(N,1)*arrQ_L1;
matQ_L0=ones(N,1)*arrQ_L0;
matQ_L1_2=ones(N,1)*arrQ_L1_2;
matQ_L2_3=ones(N,1)*arrQ_L2_3;
rho=3;
epsilon=0;

%% noise variance
sigma2_v=(arrP*(arrR.^(2)).')*N/(10^(SNR/10));

%% function hundle
func_prox_L1=@(u,gamma) prox_L1(u,gamma);
func_prox_L0=@(u,gamma) prox_L0(u,gamma);
func_prox_L1_2=@(u,gamma) prox_L1_2(u,gamma);
func_prox_L2_3=@(u,gamma) prox_L2_3(u,gamma);
func_prox_L1L2=@(u,gamma) prox_L1L2(u,gamma);

%% empirical resonstruction
arrSumSER_ADMM_SSR_L1=zeros(1,nIteration);
arrSumSER_ADMM_SSR_L0=zeros(1,nIteration);
arrSumSER_ADMM_SSR_L1_2=zeros(1,nIteration);
arrSumSER_ADMM_SSR_L2_3=zeros(1,nIteration);
arrSumSER_ADMM_SSR_L1L2=zeros(1,nIteration);
arrSumSER_PDS_SSR_L1=zeros(1,nIteration);
arrSumSER_PDS_SSR_L0=zeros(1,nIteration);
arrSumSER_PDS_SSR_L1_2=zeros(1,nIteration);
arrSumSER_PDS_SSR_L2_3=zeros(1,nIteration);
arrSumSER_PDS_SSR_L1L2=zeros(1,nIteration);
for sampleIndex=1:nSample
  disp(['sampleIndex=' num2str(sampleIndex)]);
  % unknown discrete-valued vector
  x_rand=rand(N,1);
  x=ones(N,1)*arrR(1);
  for valueIndex=2:L
    x(x_rand>=arrCDF(valueIndex-1))=arrR(valueIndex);
  end
  % measurement matrix
  A=randn(M,N);
  % additive noise vector
  v=randn(M,1)*sqrt(sigma2_v);
  % linear measurements
  y=A*x+v;

  % parameters of PDS
  beta=lambda*norm(A'*A,2);
  rho1=2/(beta+4);
  rho2=1/L;
  
  %% ADMM-SSR
  invMat=(rho*L*eye(N)+lambda*(A'*A))^(-1);
  % L1 (SOAV)
  [x_est_L1,arrSER_ADMM_SSR_L1]=ADMM_SSR(y,A,arrR,matQ_L1,invMat,lambda,rho,func_prox_L1,nIteration,epsilon,x);
  arrSumSER_ADMM_SSR_L1=arrSumSER_ADMM_SSR_L1+arrSER_ADMM_SSR_L1;
  % L0
  [x_est_L0,arrSER_ADMM_SSR_L0]=ADMM_SSR(y,A,arrR,matQ_L0,invMat,lambda,rho,func_prox_L0,nIteration,epsilon,x);
  arrSumSER_ADMM_SSR_L0=arrSumSER_ADMM_SSR_L0+arrSER_ADMM_SSR_L0;
  % L1_2
  [x_est_L1_2,arrSER_ADMM_SSR_L1_2]=ADMM_SSR(y,A,arrR,matQ_L1_2,invMat,lambda,rho,func_prox_L1_2,nIteration,epsilon,x);
  arrSumSER_ADMM_SSR_L1_2=arrSumSER_ADMM_SSR_L1_2+arrSER_ADMM_SSR_L1_2;
  % L2_3
  [x_est_L2_3,arrSER_ADMM_SSR_L2_3]=ADMM_SSR(y,A,arrR,matQ_L2_3,invMat,lambda,rho,func_prox_L2_3,nIteration,epsilon,x);
  arrSumSER_ADMM_SSR_L2_3=arrSumSER_ADMM_SSR_L2_3+arrSER_ADMM_SSR_L2_3;
  % L1L2
  [x_est_L1L2,arrSER_ADMM_SSR_L1L2]=ADMM_SSR(y,A,arrR,matQ_L1,invMat,lambda,rho,func_prox_L1L2,nIteration,epsilon,x);
  arrSumSER_ADMM_SSR_L1L2=arrSumSER_ADMM_SSR_L1L2+arrSER_ADMM_SSR_L1L2;

  %% PDS-SSR
  % L1 (SOAV)
  [x_est_L1,arrSER_PDS_SSR_L1]=PDS_SSR(y,A,arrR,matQ_L1,lambda,rho1,rho2,func_prox_L1,nIteration,epsilon,x);
  arrSumSER_PDS_SSR_L1=arrSumSER_PDS_SSR_L1+arrSER_PDS_SSR_L1;
  % L0
  [x_est_L0,arrSER_PDS_SSR_L0]=PDS_SSR(y,A,arrR,matQ_L0,lambda,rho1,rho2,func_prox_L0,nIteration,epsilon,x);
  arrSumSER_PDS_SSR_L0=arrSumSER_PDS_SSR_L0+arrSER_PDS_SSR_L0;
  % L1_2
  [x_est_L1_2,arrSER_PDS_SSR_L1_2]=PDS_SSR(y,A,arrR,matQ_L1_2,lambda,rho1,rho2,func_prox_L1_2,nIteration,epsilon,x);
  arrSumSER_PDS_SSR_L1_2=arrSumSER_PDS_SSR_L1_2+arrSER_PDS_SSR_L1_2;
  % L2_3
  [x_est_L2_3,arrSER_PDS_SSR_L2_3]=PDS_SSR(y,A,arrR,matQ_L2_3,lambda,rho1,rho2,func_prox_L2_3,nIteration,epsilon,x);
  arrSumSER_PDS_SSR_L2_3=arrSumSER_PDS_SSR_L2_3+arrSER_PDS_SSR_L2_3;
  % L1L2
  [x_est_L1L2,arrSER_PDS_SSR_L1L2]=PDS_SSR(y,A,arrR,matQ_L1,lambda,rho1,rho2,func_prox_L1L2,nIteration,epsilon,x);
  arrSumSER_PDS_SSR_L1L2=arrSumSER_PDS_SSR_L1L2+arrSER_PDS_SSR_L1L2;

end
arrSER_ADMM_SSR_L1=arrSumSER_ADMM_SSR_L1/nSample;
arrSER_ADMM_SSR_L0=arrSumSER_ADMM_SSR_L0/nSample;
arrSER_ADMM_SSR_L1_2=arrSumSER_ADMM_SSR_L1_2/nSample;
arrSER_ADMM_SSR_L2_3=arrSumSER_ADMM_SSR_L2_3/nSample;
arrSER_ADMM_SSR_L1L2=arrSumSER_ADMM_SSR_L1L2/nSample;
arrSER_PDS_SSR_L1=arrSumSER_PDS_SSR_L1/nSample;
arrSER_PDS_SSR_L0=arrSumSER_PDS_SSR_L0/nSample;
arrSER_PDS_SSR_L1_2=arrSumSER_PDS_SSR_L1_2/nSample;
arrSER_PDS_SSR_L2_3=arrSumSER_PDS_SSR_L2_3/nSample;
arrSER_PDS_SSR_L1L2=arrSumSER_PDS_SSR_L1L2/nSample;

%% Display results
close all;
nPoint=10;
arrPlotIteration=[1 (nIteration/nPoint):(nIteration/nPoint):nIteration];
figure;
h=semilogy(1:nIteration,arrSER_ADMM_SSR_L1,'-s','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
hold on;
h=semilogy(1:nIteration,arrSER_ADMM_SSR_L2_3,'-v','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(1:nIteration,arrSER_ADMM_SSR_L1_2,'-^','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(1:nIteration,arrSER_ADMM_SSR_L0,'-o','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(1:nIteration,arrSER_ADMM_SSR_L1L2,'-d','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
fig=gca;
fig.ColorOrderIndex=1;
h=semilogy(1:nIteration,arrSER_PDS_SSR_L1,'--s','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
h=semilogy(1:nIteration,arrSER_PDS_SSR_L2_3,'--v','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
h=semilogy(1:nIteration,arrSER_PDS_SSR_L1_2,'--^','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
h=semilogy(1:nIteration,arrSER_PDS_SSR_L0,'--o','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
h=semilogy(1:nIteration,arrSER_PDS_SSR_L1L2,'--d','LineWidth',1,'MarkerSize',10,'MarkerIndices',arrPlotIteration,'MarkerFaceColor','auto');
grid on;
objLegend=legend('ADMM-SSR ($\ell_{1}$)','ADMM-SSR ($\ell_{2/3}$)','ADMM-SSR ($\ell_{1/2}$)','ADMM-SSR ($\ell_{0}$)','ADMM-SSR ($\ell_{1}-\ell_{2}$)','SSR-PDS ($\ell_{1}$)','SSR-PDS ($\ell_{2/3}$)','SSR-PDS ($\ell_{1/2}$)','SSR-PDS ($\ell_{0}$)','SSR-PDS ($\ell_{1}-\ell_{2}$)');
xlabel('number of iterations');
ylabel('SER');
objLegend.Interpreter='latex';
objLegend.Location='northeast';
objLegend.FontSize=18;
fig=gca;
fig.FontSize=18;
fig.TickLabelInterpreter='latex';
fig.XLabel.Interpreter='latex';
fig.YLabel.Interpreter='latex';
axis([0 nIteration 1e-5 1]);
saveas(h, 'SSR_vsIteration.eps', 'epsc');
