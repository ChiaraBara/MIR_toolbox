%% decomposition of th Mutual Information Rate
% Y: bivariate time series (N x 2)
% m: memory (number of past samples in the embedding vectors)
% tau: vector of embedding delays (one for each series)
% base: base of the logarithm for entropy computation (2, or not pass argument to measure in bits, 0 to measure in nats)

function out=MIR_MIRdec_perm(Y,m,tau,base)

M=size(Y,2);

% CE H(Yn|Yn^m)
V_Y=MIR_SetLag([0 m],tau,ones(1,M),[0 0]);
[MyY,~]=MIR_ObsMat(Y,2,V_Y);
[~,MyYp]=sort(MyY,2); [~,MyYp]=sort(MyYp,2);
[~,MYp]=sort(MyY(:,2:m+1),2); [~,MYp]=sort(MYp,2);
Myp=MyYp(:,1);
HyY=MIR_H([Myp MYp],base);
HY=MIR_H(MYp,base);
Hy_Y=HyY-HY;

% CE H(Xn|Xn^m)
V_X=MIR_SetLag([m 0],tau,ones(1,M),[0 0]);
[MxX,~]=MIR_ObsMat(Y,1,V_X);
[~,MxXp]=sort(MxX,2); [~,MxXp]=sort(MxXp,2); 
[~,MXp]=sort(MxX(:,2:m+1),2); [~,MXp]=sort(MXp,2); 
Mxp=MxXp(:,1);
HxX=MIR_H([Mxp MXp],base);
HX=MIR_H(MXp,base);
Hx_X=HxX-HX;
 
HyXY=MIR_H([Myp MYp MXp],base);
HXY=MIR_H([MXp MYp],base);
Hy_XY=HyXY-HXY;

% CE H(Xn|Xn^m,Yn^m) 
HxXY=MIR_H([Mxp MXp MYp],base);
Hx_XY=HxXY-HXY;

% CE H(Yn|Xn,Xn^m,Yn^m)
HyxXY=MIR_H([Myp Mxp MYp MXp],base);
Hy_xXY=HyxXY-HxXY;

% CE H(Xn,Yn|Xn^m,Yn^m)
Hxy_XY=HyxXY-HXY;

% Transfer entropy X->Y (1->2)
T_XY=Hy_Y-Hy_XY;

% Transfer entropy Y->X (2->1)
T_YX=Hx_X-Hx_XY;

% instantaneous information shared
I_YoX=Hy_XY-Hy_xXY;

% verification instantaneous dependence
HxyXY=MIR_H([Mxp Myp MXp MYp],base);
Hx_yXY=HxyXY-HyXY;
I_XoY=Hx_XY-Hx_yXY;

% Mutual Information Rate
I_XY2=T_XY+T_YX+I_XoY;

% verification of MIR
I_XY=Hx_X+Hy_Y-Hxy_XY;

%% output structure
out.Hy_Y=Hy_Y;
out.Hx_X=Hx_X;
out.Hxy_XY=Hxy_XY;
out.I_XY=I_XY;

out.T_XY=T_XY;
out.T_YX=T_YX;
out.I_XoY=I_XoY;
out.I_YoX=I_YoX;
out.I_XY2=I_XY2;

out.Hx_XY=Hx_XY;
out.Hy_XY=Hy_XY;

out.MxX=MxX;
out.MyY=MyY;