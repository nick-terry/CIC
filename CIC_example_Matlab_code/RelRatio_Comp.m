%% 7/9/2015 Reference codes: C:\Joun\Box\Research\Implementation\Code\Flux_Rel_Ratio.m
%% Compute relative ratio

% CE-AIS-GM  vs. EMCE:  b = 1.5
N_T = 8700;
P = 0.082911;
SE = [0.001145; 0.000506];
N_T_CMC = P.*(1-P)./(SE.^2);
Rel_Ratio = N_T./N_T_CMC*100;

% CE-AIS-GM  vs. EMCE:  b = 2
N_T = 8700;
P = 0.030173;
SE = [0.000526; 0.000213];
N_T_CMC = P.*(1-P)./(SE.^2);
Rel_Ratio = N_T./N_T_CMC*100;

% CE-AIS-GM  vs. EMCE:  b = 2
N_T = 8700;
P = 0.008910;
SE = [0.000211; 0.000099];
N_T_CMC = P.*(1-P)./(SE.^2);
Rel_Ratio = N_T./N_T_CMC*100;



