% Goal: Illustrate how the EM algorithm for CE minimization works 
% Created on 10/15/2018 

%% K = 3 using EM algorithm 
clear
K_fix = 3;
%% Setup
n_exp = 1;
randseed = 1000;
rng(randseed);

%% Eq (19) and setup in p. 39 LB of KS(2013)
%%Table 1 example: bb = 5; kk = 0.5; ee = 0.1;
bb = 1.5; kk = 0.1; ee = 0; %Table 2 example (instead of bb = 5)
H = @(x) (bb - x(:,2) - kk.*(x(:,1) - ee).^2) <= 0;
fu = @(x) mvnpdf(x,zeros(1,2),eye(2));

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
n_steps = 1; %6;
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

%     % Evaluate the simulation output, H(x), and the likelihood ratio W(x).
%     hw = pdf(GMM{t},X_new);
%     X = [X; X_new]; Hx_new = H(X_new); Wx_new = fu(X_new)./hw; %#ok<AGROW>
%     Hx = [Hx; Hx_new];  Wx = [Wx; Wx_new]; %#ok<AGROW>
%     Hx_Wx = [Hx_Wx; Hx_new.*Wx_new]; %#ok<AGROW>
%     % Current POE estimate
%     current_est = mean(Hx_Wx((n_per_initstep+1):end)); 
end

% %% Final sampling and estimation
% 	fprintf('t=%d (from i=%d)\n', t+1, i_exp) 
%     [GMM_final, best_models_by_k_final, CIC_final, nCEs_for_best_models_final, k_min] = EM_Algo_fixed_K( X, Hx_Wx, current_est, k_min, k_max ); 
%     % Sampling from the last GMM
%     X_new = random(GMM_final,n_final);
%     
%     % Evaluate the simulation output, H(x), and the likelihood ratio W(x).
%     hw = pdf(GMM_final,X_new);
%     X = [X; X_new]; Hx_new = H(X_new); Wx_new = fu(X_new)./hw; 
%     Hx = [Hx; Hx_new];  Wx = [Wx; Wx_new]; 
%     Hx_Wx = [Hx_Wx; Hx_new.*Wx_new]; 
% 
%     %Final estimation and density
%     final_est =  mean(Hx_Wx((n_per_initstep+1):end)); %Exclude the initial samples.    mean(Hx_Wx); 
%     final_density = GMM_final;
%     model_info = {best_models_by_k_final, CIC_final, nCEs_for_best_models_final, Hx_Wx, GMM, best_models_by_k, CIC, nCEs_for_best_models};
% %Elapsed time is 2090.871826 seconds. %Ex1.

%% Plotting for the converged GMM
% Plot prep
half_display_range = 4;
x = -half_display_range:0.025:half_display_range;
y =  half_display_range:-0.025:-half_display_range;
x_len = length(x);
y_len = length(y);

x_long = repmat(x',[y_len,1]);
y_long = repelem(y',x_len);
xy_long = [x_long,y_long];

t=1;
GM = GMM{t}; %final_density; 
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
hold on
xlim([-half_display_range, half_display_range])
ylim([-half_display_range, half_display_range])

scatter(X(:,1),X(:,2),2,'MarkerEdgeColor',[218,165,32]/255,'MarkerFaceColor',[218,165,32]/255)%https://www.rapidtables.com/web/color/RGB_Color.html

%ezcontour(@(x,y)pdf(GM,[x y]),[-half_display_range half_display_range],[-half_display_range half_display_range])

MVN_pdf=cell(1,K_fix);
for k =1:K_fix
    MVN_pdf{k} = @(x,y)mvnpdf([x y],GM.mu(k,:),GM.Sigma(:,:,k));    
    %Plot of proportion-weighted densities
    fcontour(@(x,y) GM.ComponentProportion(k).*MVN_pdf{k}(x,y),[-half_display_range half_display_range -half_display_range half_display_range])   
end
hold off
save2pdf('EM_iteration_coverged', gcf, 600);

