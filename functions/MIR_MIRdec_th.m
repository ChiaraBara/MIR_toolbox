%% Theoretical value of MIR decomposition for a bivariate VAR process
% performs linear regression of the present state of N target processes from the past states of M driver processes
% i: indexes of the first process, X (DRIVER)
% j: indexes of the second process, Y (TARGET)
% p: model order of the processes
% Am, Su: theoretical VAR model parameters 

function ret = MIR_MIRdec_th(Am,Su,p,i,j)

[SigmaS,SigmaS_S] = MIR_LinReg(Am,Su,p,[i j],[i j]); % bivariate model, [Ypresent Xpresent] given [Ypast Xpast]
[~,SigmaX_XY] = MIR_LinReg(Am,Su,p,i,[i j]); % full model, Ypresent given [Ypast Xpast]
[~,SigmaY_XY] = MIR_LinReg(Am,Su,p,j,[i j]); % full model, Xpresent given [Ypast Xpast]

[SigmaY,SigmaY_Y] = MIR_LinReg(Am,Su,p,j,j); % restricted model, Ypresent given Ypast
[SigmaX,SigmaX_X] = MIR_LinReg(Am,Su,p,i,i); % restricted model, Xpresent given Xpast

Hx_X = 0.5*log(2*pi*exp(1)*SigmaX_X/SigmaX);  % CE of X
Hy_Y = 0.5*log(2*pi*exp(1)*SigmaY_Y/SigmaY);  % CE of Y
SigmaS_S(1,1) = SigmaS_S(1,1)/SigmaS(1,1); SigmaS_S(2,2) = SigmaS_S(2,2)/SigmaS(2,2);
SigmaS_S(1,2) = SigmaS_S(1,2)/(sqrt(SigmaS(1,1))*sqrt(SigmaS(2,2))); SigmaS_S(2,1) = SigmaS_S(2,1)/(sqrt(SigmaS(1,1))*sqrt(SigmaS(2,2)));
Hxy_XY = 0.5*log((2*pi*exp(1))^2*det(SigmaS_S)); % joint CE of X and Y
I_XY = Hx_X + Hy_Y - Hxy_XY; % total dependence MIR

T_XY=0.5*log(SigmaY_Y/SigmaY_XY); % TE from X to Y
T_YX=0.5*log(SigmaX_X/SigmaX_XY); % TE from Y to X
I_XoY=0.5*log((SigmaX_XY*SigmaY_XY)/(SigmaX*SigmaY*det(SigmaS_S))); % instantaneous causality of X and Y
I_XY2=T_XY+T_YX+I_XoY; % total dependence MIR

%% output

ret.I_XY=I_XY; 
ret.Hx_X=Hx_X; 
ret.Hy_Y=Hy_Y; 
ret.Hxy_XY = Hxy_XY;

ret.I_XY2=I_XY2; 
ret.T_XY=T_XY; 
ret.T_YX=T_YX; 
ret.I_XoY=I_XoY; 

end
