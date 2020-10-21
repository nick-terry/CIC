%10/17/2018: Added a line to newly create
%'CIC-IS_optimal_density_bb15.pdf'.
%2/3/2016: Density images for presentation purpose. 

% Draw the contour plot
load('Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat')

%% Optimal density plot
addpath('./mcint')
n = 1e16; %1e6; % more points = more accuracy. Relative error goes as 1/sqrt(n)
int_mcint = @(x) (H(x').*fu(x'))'; % @(x) (f(x').* s(x'))'; %Transposes are necessary to be compatible with mcint().
opt.noerror = 0;
opt.tol = 1e-7; %2e-4;
opt.time = 60; %in seconds
opt.warnOff = 0;
opt.mcintPar = [int_mcint]; %[];
h.b = [ [-Inf, Inf];[-Inf, Inf] ];
h.cond = {};
h.funcs = {@integrand_mcint}; %integrand_mcint.m  function file is created. %{(@(x) f(x').* sqrt(s_meta(x')./N_T + (1-1/N_T).*s_meta(x').^2))}; DOESN't work this way because an additional parameter needs to be passed.
mcint_output = mcint(h, n, opt);  %Note: the error is contained in mcint_output(1,1,2);  calculation code: getError(s,points,V) in mcint.m
norm_constant_IS_pdf = mcint_output(1,1,1);

f_opt = @(x,y) H([x,y]).*fu([x,y])./norm_constant_IS_pdf;

% Plot prep
x = -4:0.025:4;
y =  4:-0.025:-4;
x_len = length(x);
y_len = length(y);

x_long = repmat(x',[y_len,1]);
y_long = repelem(y',x_len);
xy_long = [x_long,y_long];
z_long = H(xy_long).*fu(xy_long)./norm_constant_IS_pdf;
z = reshape(z_long, [x_len, y_len])';


figure
colormap(gray) %bone
contourf(x,y,-z)
hold on
y_ref = (bb - kk.*(x - ee).^2);
plot(x,y_ref, 'r--')
hold off
xlabel('x_1'); ylabel('x_2')
save2pdf('CIC-IS_optimal_density_bb15', gcf, 600); 
% save2pdf('Ex1_optimal_density_02032016', gcf, 600); %300); %resolution should be 300 dpi in both black and white and color cases.
% save2pdf('Ex1_optimal_density_bb25_02072016', gcf, 600); 
%% 
% i = 8;
% %for i = 1:10
%     %GM = final_density_list_CEM{i};  
%     %GM = final_density_list_KS{1};
%     GM = model_info{1,5}{1,5}{1,6}; %GM = model_info{1,4}{1,1}{1,12}; %K=12 %model_info{1,4}{1,1}{1,3}; %K=3 
% 
%     z_GM_long = pdf(GM,xy_long);
%     z_GM = reshape(z_GM_long, [x_len, y_len])';
% 
%     figure
%     colormap(gray)
%     contourf(x,y,-z_GM)
% 
%     hold on
%     y_ref = (bb - kk.*(x - ee).^2);
%     plot(x,y_ref, 'r--')
%     hold off
% %end
% xlabel('x_1'); ylabel('x_2')
% % save2pdf('Ex1_EMCE_density_02032016', gcf, 600); %300); %resolution should be 300 dpi in both black and white and color cases.
% 
% GM.NumComponents

%% K = 5 using EM algorithm 
%clear
K_fix = 30; %5;
%% Setup
n_exp = 1;
randseed = 100;
rng(randseed);

%% Eq (19) and setup in p. 39 LB of KS(2013)
%%Table 1 example: bb = 5; kk = 0.5; ee = 0.1;
% bb = 2.5; kk = 0.1; ee = 0; %Table 2 example (instead of bb = 5)
% H = @(x) (bb - x(:,2) - kk.*(x(:,1) - ee).^2) <= 0;
% fu = @(x) mvnpdf(x,zeros(1,2),eye(2));

%% Initial distribution
n_comp = 30;
Xdim = 2;
c_spread = 3;

mu0_list = cell(1,n_exp);
for i = 1:n_exp
    mu0_list{i} = mvnrnd(zeros(n_comp,Xdim),eye(Xdim));
end

%mu0 = mvnrnd(zeros(n_comp,Xdim),eye(Xdim));
sig0 = c_spread*eye(Xdim);
w0 = ones(1,n_comp)./n_comp;


i_exp = 1;
mu0 = mu0_list{i_exp};

%% CEM

%% Setup used for Table 2 in KS(2013)
n_steps = 6;
n_per_initstep = 10^3; 
n_per_step = 10^3; 
n_final = 1700; % the case corresponding to c.o.v. = 3%

%% t = 0
initGM = gmdistribution(mu0, sig0, w0);
X = random(initGM, n_per_initstep);

hw = pdf(initGM,X);
Hx = H(X);
Wx = fu(X)./hw;
Hx_Wx = Hx.*Wx;
current_est = mean(Hx_Wx); 

%% t = 1 through n_steps
GMM = cell(1,n_steps);
best_models_by_k = cell(1,n_steps);
CIC = cell(1,n_steps);
nCEs_for_best_models = cell(1,n_steps);

k_min = K_fix; k_max = K_fix;
for t = 1:n_steps
    fprintf('t=%d (from i=%d)\n', t, i_exp)        
    % EM Algorithm    
    [GMM{t}, best_models_by_k{t}, CIC{t}, nCEs_for_best_models{t}, k_min] = EM_Algo_fixed_K( X, Hx_Wx, current_est, k_min, k_max ); 
    % Sampling from the last GMM
    X_new = random(GMM{t},n_per_step);

    % Evaluate the simulation output, H(x), and the likelihood ratio W(x).
    hw = pdf(GMM{t},X_new);
    X = [X; X_new]; Hx_new = H(X_new); Wx_new = fu(X_new)./hw; %#ok<AGROW>
    Hx = [Hx; Hx_new];  Wx = [Wx; Wx_new]; %#ok<AGROW>
    Hx_Wx = [Hx_Wx; Hx_new.*Wx_new]; %#ok<AGROW>
    % Current POE estimate
    current_est = mean(Hx_Wx((n_per_initstep+1):end)); 
end

%% Final sampling and estimation
	fprintf('t=%d (from i=%d)\n', t+1, i_exp) 
    [GMM_final, best_models_by_k_final, CIC_final, nCEs_for_best_models_final, k_min] = EM_Algo_fixed_K( X, Hx_Wx, current_est, k_min, k_max ); 
    % Sampling from the last GMM
    X_new = random(GMM_final,n_final);
    
    % Evaluate the simulation output, H(x), and the likelihood ratio W(x).
    hw = pdf(GMM_final,X_new);
    X = [X; X_new]; Hx_new = H(X_new); Wx_new = fu(X_new)./hw; 
    Hx = [Hx; Hx_new];  Wx = [Wx; Wx_new]; 
    Hx_Wx = [Hx_Wx; Hx_new.*Wx_new]; 

    %Final estimation and density
    final_est =  mean(Hx_Wx((n_per_initstep+1):end)); %Exclude the initial samples.    mean(Hx_Wx); 
    final_density = GMM_final;
    model_info = {best_models_by_k_final, CIC_final, nCEs_for_best_models_final, Hx_Wx, GMM, best_models_by_k, CIC, nCEs_for_best_models};
%Elapsed time is 2090.871826 seconds. %Ex1.

%% Plotting
GM = final_density; 
%GM = GMM{1}; %First Step: n = 2000
 z_GM_long = pdf(GM,xy_long);
    z_GM = reshape(z_GM_long, [x_len, y_len])';

    figure
    colormap(gray)
    contourf(x,y,-z_GM)

    hold on
    y_ref = (bb - kk.*(x - ee).^2);
    plot(x,y_ref, 'r--')
    hold off

xlabel('x_1'); ylabel('x_2')
save2pdf('Ex1_EMCE_density_K1_maxiter10', gcf, 600);
% save2pdf('Ex1_EMCE_density_K30_maxiter10_singularity_threshold_1e10_FirstStep', gcf, 600);
% save2pdf('Ex1_EMCE_density_K30_maxiter10_singularity_threshold_1e10', gcf, 600);


%% KS density plot (bb = 1.5)
load('Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat')
GM = final_density_list_KS{1};
x = -4:0.025:4;
y =  4:-0.025:-4;
 z_GM_long = pdf(GM,xy_long);
    z_GM = reshape(z_GM_long, [x_len, y_len])';

    figure
    colormap(gray)
    contourf(x,y,-z_GM)

    hold on
    y_ref = (bb - kk.*(x - ee).^2);
    plot(x,y_ref, 'r--')
    hold off

xlabel('x_1'); ylabel('x_2')

