clear all; clc; %close all; 

addpath([pwd,'/functions/']);

%%% simulation parameters
coup=[0:0.05:1]';
M=2;
mth = 4;
N = 300; % time series length

%%%% estimation parameters
%%% linear estimator
p=3; %model order
q=20; %number of lags for estimation of correlations
%%% knn estimator
m_knn=3;
k=10;
%%% binning estimator
m_bin=1; %number of past lags of Markov processes
b=4; % n. of bins
%%% permutation estimator
m_perm=3; %number of past lags of Markov processes
%%% other par
base=2; %2 for entropy in bits, 0 for entropy in nats
tau = [1 1];

%%% computation 

for ic=1:numel(coup)

    par.poles{1}=([0.2*coup(ic) 0.03; 0.8*coup(ic) 0.1]); % Oscillations RR X
    par.poles{2}=([0.9 0.3]); % Oscillation RESP y
    par.Su=[1 1];
    M=2;
    par.coup=[2 1 2 1-coup(ic)];
    [Am,Su,Ak]=theoreticalVAR(M,par); %% VAR parameters
    
    %%% theoretical MIR
    ret = MIR_MIRdec_th(Am',Su,mth,1,2);
    
    HX(ic) = ret.Hx_X;
    HY(ic) = ret.Hy_Y;
    HXY(ic) = ret.Hxy_XY;
    TXY(ic) = ret.T_XY;
    TYX(ic) = ret.T_YX;
    IXoY(ic) =ret.I_XoY;
    IXY(ic) = ret.I_XY;
    IXY2(ic) = ret.I_XY2;
    
    Un = mvnrnd(zeros(1,M),Su,N);
    Yn = var_filter(Am',Un); % realization
    
    Yn = zscore(Yn);
    
    %%% linear estimation of MIR
    out=MIR_MIRdec_lin(Yn,p,tau,q);
    HX_lin(ic) = out.Hx_X;
    HY_lin(ic) = out.Hy_Y;
    HXY_lin(ic) = out.Hxy_XY;
    TXY_lin(ic) = out.T_XY;
    TYX_lin(ic) = out.T_YX;
    IXoY_lin(ic) = out.I_XoY;
    IXY_lin(ic) = out.I_XY;
    IXY2_lin(ic) = out.I_XY2;
    
    %%% knn estimation of MIR
    out=MIR_MIRdec_knn(Yn,m_knn,tau,k);
    HX_knn(ic) = out.Hx_X;
    HY_knn(ic) = out.Hy_Y;
    HXY_knn(ic) = out.Hxy_XY;
    TXY_knn(ic) = out.T_XY;
    TYX_knn(ic) = out.T_YX;
    IXoY_knn(ic) = out.I_XoY;
    IXY_knn(ic) = out.I_XY;
    IXY2_knn(ic) = out.I_XY2;
    
    %%% bin estimation of MIR
    out=MIR_MIRdec_bin(Yn,b,m_bin,tau,base);
    HX_bin(ic) = out.Hx_X;
    HY_bin(ic) = out.Hy_Y;
    HXY_bin(ic) = out.Hxy_XY;
    TXY_bin(ic) = out.T_XY;
    TYX_bin(ic) = out.T_YX;
    IXoY_bin(ic) = out.I_XoY;
    IXY_bin(ic) = out.I_XY;
    IXY2_bin(ic) = out.I_XY2;
    
    %%% perm estimation of MIR
    out=MIR_MIRdec_perm(Yn,m_perm,tau,base);
    HX_perm(ic) = out.Hx_X;
    HY_perm(ic) = out.Hy_Y;
    HXY_perm(ic) = out.Hxy_XY;
    TXY_perm(ic) = out.T_XY;
    TYX_perm(ic) = out.T_YX;
    IXoY_perm(ic) = out.I_XoY;
    IXY_perm(ic) = out.I_XY;
    IXY2_perm(ic) = out.I_XY2;
   
end

%% MIR PLOT

figure;

subplot(2,5,1);
plot(coup,HX,'k'); ylabel('H_{X}');
hold on; plot(coup,HX_lin);
hold on; plot(coup,HX_knn);
hold on; plot(coup,HX_bin);
hold on; plot(coup,HX_perm);

subplot(2,5,2);
plot(coup,HY,'k'); ylabel('H_{Y}');
hold on; plot(coup,HY_lin);
hold on; plot(coup,HY_knn);
hold on; plot(coup,HY_bin);
hold on; plot(coup,HY_perm);

subplot(2,5,3);
plot(coup,HXY,'k'); ylabel('H_{X,Y}');
hold on; plot(coup,HXY_lin);
hold on; plot(coup,HXY_knn);
hold on; plot(coup,HXY_bin);
hold on; plot(coup,HXY_perm);

subplot(2,5,6);
plot(coup,TXY,'k'); ylabel('T_{X \rightarrow Y}');
hold on; plot(coup,TXY_lin);
hold on; plot(coup,TXY_knn);
hold on; plot(coup,TXY_bin);
hold on; plot(coup,TXY_perm);

subplot(2,5,7);
plot(coup,TYX,'k'); ylabel('T_{Y \rightarrow X}');
hold on; plot(coup,TYX_lin);
hold on; plot(coup,TYX_knn);
hold on; plot(coup,TYX_bin);
hold on; plot(coup,TYX_perm);

subplot(2,5,8);
plot(coup,IXoY,'k'); ylabel('I_{X \cdot Y}');
hold on; plot(coup,IXoY_lin);
hold on; plot(coup,IXoY_knn);
hold on; plot(coup,IXoY_bin);
hold on; plot(coup,IXoY_perm);

subplot(2,5,[4 9]);
plot(coup,IXY,'k'); ylabel('I_{X,Y} = H_{X}+H_{Y}-H_{X,Y}');
hold on; plot(coup,IXY_lin);
hold on; plot(coup,IXY_knn);
hold on; plot(coup,IXY_bin);
hold on; plot(coup,IXY_perm);

subplot(2,5,[5 10]);
plot(coup,IXY2,'k'); ylabel('I_{X,Y} = T_{X \rightarrow Y}+T_{Y \rightarrow X}+I_{X \cdot Y}');
hold on; plot(coup,IXY2_lin);
hold on; plot(coup,IXY2_knn);
hold on; plot(coup,IXY2_bin);
hold on; plot(coup,IXY2_perm);

legend({'th','lin','knn','bin','perm'});