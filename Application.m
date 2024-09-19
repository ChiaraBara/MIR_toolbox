%% applies MIR decomposition on exemplary RR-SAP time series
%%% compares binning, permutation and linear approaches
clear; close all; clc;

addpath([pwd,'/functions/']);

%%%% parameters

%%% binning estimator
m_bin=1; %number of past lags of Markov processes
b=4; % n. of bins

%%% permutation estimator
m_perm=3; %number of past lags of Markov processes

%%% linear estimator
p=3; %model order
q=20; %number of lags for estimation of correlations

%%% knn estimator
m_knn=3;
k=10;

%%% other par
base=2; %2 for entropy in bits, 0 for entropy in nats
pfilter = 0.94;

%% open data
load('data_RR_SAP.mat'); % RR->column1(X); SAP->column2(Y)
Y=data_RR_SAP; % original series
Y=zscore(Y); % normalization (not necessary for bin and perm)

[N,M]=size(Y);

%% estimation of MIR decomposition
tau=ones(1,M); %embedding lag always unitary

%%% linear approach
out_lin = MIR_MIRdec_lin(Y,p,tau,q);
l_I_XY=out_lin.I_XY;
l_Hx_X=out_lin.Hx_X;
l_Hy_Y=out_lin.Hy_Y;
l_Hxy_XY=out_lin.Hxy_XY;
l_T_XY=out_lin.T_XY;
l_T_YX=out_lin.T_YX;
l_I_XoY=out_lin.I_XoY;

%%% nearest-neighbor approach
out_knn = MIR_MIRdec_knn(Y,m_knn,tau,k);
k_I_XY=out_knn.I_XY;
k_Hx_X=out_knn.Hx_X;
k_Hy_Y=out_knn.Hy_Y;
k_Hxy_XY=out_knn.Hxy_XY;
k_T_XY=out_knn.T_XY;
k_T_YX=out_knn.T_YX;
k_I_XoY=out_knn.I_XoY;

%%% binning approach
out_bin=MIR_MIRdec_bin(Y,b,m_bin,tau,base);
b_I_XY=out_bin.I_XY;
b_Hx_X=out_bin.Hx_X;
b_Hy_Y=out_bin.Hy_Y;
b_Hxy_XY=out_bin.Hxy_XY;
b_T_XY=out_bin.T_XY;
b_T_YX=out_bin.T_YX;
b_I_XoY=out_bin.I_XoY;

%%% permutation approach
out_perm=MIR_MIRdec_perm(Y,m_perm,tau,base);
p_I_XY=out_perm.I_XY;
p_Hx_X=out_perm.Hx_X;
p_Hy_Y=out_perm.Hy_Y;
p_Hxy_XY=out_perm.Hxy_XY;
p_T_XY=out_perm.T_XY;
p_T_YX=out_perm.T_YX;
p_I_XoY=out_perm.I_XoY;

%% display

disp('Estimated values of MIR:');
disp(['Lin: ', num2str(l_I_XY),' nats']);
disp(['Knn: ', num2str(k_I_XY),' nats']);
disp(['Bin: ', num2str(b_I_XY),' bits']);
disp(['Perm: ', num2str(p_I_XY),' bits']);