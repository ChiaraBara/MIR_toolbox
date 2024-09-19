%% decomposition of th Mutual Information Rate
% Y: bivariate time series (N x 2)
% b: number of quantization bins
% m: memory (number of past samples in the embedding vectors)
% tau: vector of embedding delays (one for each series)
% base: base of the logarithm for entropy computation (2, or not pass argument to measure in bits, 0 to measure in nats)

function out=MIR_MIRdec_bin(Y,b,m,tau,base)

M=size(Y,2);
for im=1:M
    Yq(:,im)=MIR_quantization(Y(:,im),b);
end

% CE H(Yn|Yn^m)
V_Y=MIR_SetLag([0 m],tau,ones(1,M),[0 0]);
[MyY,MY]=MIR_ObsMat(Yq,2,V_Y);
HyY=MIR_H(MyY,base);
HY=MIR_H(MY,base);
Hy_Y=HyY-HY;

% CE H(Yn|Xn^m,Yn^m)
V_XY=MIR_SetLag([m m],tau,ones(1,M),[0 0]);
[MyXY,MXY]=MIR_ObsMat(Yq,2,V_XY);
HyXY=MIR_H(MyXY,base);
HXY=MIR_H(MXY,base);
Hy_XY=HyXY-HXY;

% Transfer entropy X->Y (1->2)
T_XY=Hy_Y-Hy_XY;

% CE H(Xn|Xn^m)
V_X=MIR_SetLag([m 0],tau,ones(1,M),[0 0]);
[MxX,MX]=MIR_ObsMat(Yq,1,V_X);
HxX=MIR_H(MxX,base);
HX=MIR_H(MX,base);
Hx_X=HxX-HX;

% CE H(Xn|Xn^m,Yn^m)
MxXY=MIR_ObsMat(Yq,1,V_XY);
HxXY=MIR_H(MxXY,base);
Hx_XY=HxXY-HXY;

% Transfer entropy Y->X (2->1)
T_YX=Hx_X-Hx_XY;

% CE H(Yn|Xn,Xn^m,Yn^m)
V_X0Y=MIR_SetLag([m m],tau,ones(1,M),[1 0]);
MyxXY=MIR_ObsMat(Yq,2,V_X0Y);
HyxXY=MIR_H(MyxXY,base);
Hy_xXY=HyxXY-HxXY;

% instantaneous information shared
I_YoX=Hy_XY-Hy_xXY;

% CE H(Xn,Yn|Xn^m,Yn^m)
Hxy_XY=HyxXY-HXY;

% verification instantaneous dependence
V_XY0=MIR_SetLag([m m],tau,ones(1,M),[0 1]);
MxyXY=MIR_ObsMat(Yq,1,V_XY0);
HxyXY=MIR_H(MxyXY,base);
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
out.MxXY=MxXY;
out.MyXY=MyXY;