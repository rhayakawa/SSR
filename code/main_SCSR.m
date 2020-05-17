%%---------------------------------------------------------------------------------------------------------
% Complex discrete-valued vector reconstruction via sum of complex sparse regularizer (SCSR) optimization
%
% Author:
%   Ryo Hayakawa
% Article:
%   Ryo Hayakawa and Kazunori Hayashi,
%   "Discrete-Valued Vector Reconstruction by Optimization with Sum of Sparse Regularizers,"
%   in Proc. EUSIPCO, Sept. 2019. 
%%---------------------------------------------------------------------------------------------------------

clear;
addpath("myfunctions");

%% problem setting
N=200;
Delta=0.8;
M=round(N*Delta);
arrSNR=0:2.5:30;
nSymbolVector=10;
nIteration=300;
nSample=10;

% distribution of unknown vector
L=5;
p0=0.6;
arrP=[p0 (1-p0)/4 (1-p0)/4 (1-p0)/4 (1-p0)/4];
arrC=[0 1+1j -1+1j -1-1j 1-1j];

% cumulative distribution
L=length(arrC);
matOne=ones(L,L);
arrCDF=arrP*triu(matOne);

% parameters of the proposed algorithm
lambda=0.05;
matQ=ones(N,1)*arrP;
rho=3;
epsilon=0;

% function hundles for proximity operators
func_prox_L1=@(u,gamma) prox_L1(u,gamma);
func_prox_L0=@(u,gamma) prox_L0(u,gamma);
func_prox_L1_2=@(u,gamma) prox_L1_2(u,gamma);
func_prox_L2_3=@(u,gamma) prox_L2_3(u,gamma);
func_prox_L1L2=@(u,gamma) prox_L1L2(u,gamma);

%% resonstruction
arrSumSERcurve_ADMM_SCSR_L1=zeros(1,length(arrSNR));
arrSumSERcurve_ADMM_SCSR_L0=zeros(1,length(arrSNR));
arrSumSERcurve_ADMM_SCSR_L1_2=zeros(1,length(arrSNR));
arrSumSERcurve_ADMM_SCSR_L2_3=zeros(1,length(arrSNR));
arrSumSERcurve_ADMM_SCSR_L1L2=zeros(1,length(arrSNR));
for sampleIndex=1:nSample
  % measurement matrix
  A=(randn(M,N)+1j*randn(M,N))/sqrt(2);

  % inverse matrix for ADMM
  invMat=(rho*L*eye(N)+lambda*(A'*A))^(-1);
  
  disp(['sampleIndex=' num2str(sampleIndex)]);
  for SNRIndex=1:length(arrSNR)
    SNR=arrSNR(SNRIndex);
    disp(['  SNR=' num2str(SNR)]);
    % noise variance
    sigma2_v=N*2*(1-p0)/(10^(SNR/10));

    for symbolVectorIndex=1:nSymbolVector
      % unknown discrete-valued vector
      x_rand=rand(N,1);
      x=ones(N,1)*arrC(1);
      for valueIndex=2:L
        x(x_rand>=arrCDF(valueIndex-1))=arrC(valueIndex);
      end
      % additive noise vector
      v=(randn(M,1)+1j*randn(M,1))/sqrt(2)*sqrt(sigma2_v);
      % linear measurements
      y=A*x+v;
      
      %% ADMM-SCSR
      % L1
      [x_est_L1,arrSER_ADMM_SCSR_L1]=ADMM_SCSR(y,A,arrC,matQ,invMat,lambda,rho,func_prox_L1,nIteration,epsilon,x);
      arrSumSERcurve_ADMM_SCSR_L1(SNRIndex)=arrSumSERcurve_ADMM_SCSR_L1(SNRIndex)+nnz(quantize_comp(x_est_L1,arrC)-x)/N;
      % L0
      [x_est_L0,arrSER_ADMM_SCSR_L0]=ADMM_SCSR(y,A,arrC,matQ,invMat,lambda,rho,func_prox_L0,nIteration,epsilon,x);
      arrSumSERcurve_ADMM_SCSR_L0(SNRIndex)=arrSumSERcurve_ADMM_SCSR_L0(SNRIndex)+nnz(quantize_comp(x_est_L0,arrC)-x)/N;
      % L1_2
      [x_est_L1_2,arrSER_ADMM_SCSR_L1_2]=ADMM_SCSR(y,A,arrC,matQ,invMat,lambda,rho,func_prox_L1_2,nIteration,epsilon,x);
      arrSumSERcurve_ADMM_SCSR_L1_2(SNRIndex)=arrSumSERcurve_ADMM_SCSR_L1_2(SNRIndex)+nnz(quantize_comp(x_est_L1_2,arrC)-x)/N;
      % L2_3
      [x_est_L2_3,arrSER_ADMM_SCSR_L2_3]=ADMM_SCSR(y,A,arrC,matQ,invMat,lambda,rho,func_prox_L2_3,nIteration,epsilon,x);
      arrSumSERcurve_ADMM_SCSR_L2_3(SNRIndex)=arrSumSERcurve_ADMM_SCSR_L2_3(SNRIndex)+nnz(quantize_comp(x_est_L2_3,arrC)-x)/N;
      % L1L2
      [x_est_L1L2,arrSER_ADMM_SCSR_L1L2]=ADMM_SCSR(y,A,arrC,matQ,invMat,lambda,rho,func_prox_L1L2,nIteration,epsilon,x);
      arrSumSERcurve_ADMM_SCSR_L1L2(SNRIndex)=arrSumSERcurve_ADMM_SCSR_L1L2(SNRIndex)+nnz(quantize_comp(x_est_L1L2,arrC)-x)/N;

    end
  end
end
arrSERcurve_ADMM_SCSR_L1=arrSumSERcurve_ADMM_SCSR_L1/nSample/nSymbolVector;
arrSERcurve_ADMM_SCSR_L0=arrSumSERcurve_ADMM_SCSR_L0/nSample/nSymbolVector;
arrSERcurve_ADMM_SCSR_L1_2=arrSumSERcurve_ADMM_SCSR_L1_2/nSample/nSymbolVector;
arrSERcurve_ADMM_SCSR_L2_3=arrSumSERcurve_ADMM_SCSR_L2_3/nSample/nSymbolVector;
arrSERcurve_ADMM_SCSR_L1L2=arrSumSERcurve_ADMM_SCSR_L1L2/nSample/nSymbolVector;

%% Display results
close all;
figure;
h=semilogy(arrSNR,arrSERcurve_ADMM_SCSR_L1,'-s','LineWidth',1,'MarkerSize',10,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
hold on;
h=semilogy(arrSNR,arrSERcurve_ADMM_SCSR_L2_3,'-v','LineWidth',1,'MarkerSize',10,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(arrSNR,arrSERcurve_ADMM_SCSR_L1_2,'-^','LineWidth',1,'MarkerSize',10,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(arrSNR,arrSERcurve_ADMM_SCSR_L0,'-o','LineWidth',1,'MarkerSize',10,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
h=semilogy(arrSNR,arrSERcurve_ADMM_SCSR_L1L2,'-d','LineWidth',1,'MarkerSize',10,'MarkerFaceColor','auto');
set(h, 'MarkerFaceColor', get(h,'Color'));
grid on;
objLegend=legend('ADMM-SCSR ($\ell_{1}$)','ADMM-SCSR ($\ell_{2/3}$)','ADMM-SCSR ($\ell_{1/2}$)','ADMM-SCSR ($\ell_{0}$)','ADMM-SCSR ($\ell_{1}-\ell_{2}$)');
xlabel('SNR (dB)');
ylabel('SER');
objLegend.Interpreter='latex';
objLegend.Location='northeast';
objLegend.FontSize=18;
fig=gca;
fig.FontSize=18;
fig.TickLabelInterpreter='latex';
fig.XLabel.Interpreter='latex';
fig.YLabel.Interpreter='latex';
axis([min(arrSNR) max(arrSNR) 1e-5 1]);
saveas(h, 'SCSR.eps', 'epsc');
