function [ final_est, final_density, model_info ] = CEM( w0, mu0, sig0, H, fu, i_exp )
% Kurtz and Song (2013) method
%   w0: initial weights : n_comp by 1, where n_comp is the number of component densities.
%   mu0: initial mean vectors : n_comp by Xdim, where Xdim is the dimension of X.
%   sig0: initial covariances : n_comp by n_comp

%% Setup used for Table 2 in KS(2013)
n_steps = 6;
n_per_initstep = 10^4; 
n_per_step = 10^4; 
n_final = 2*10^4; % the case corresponding to c.o.v. = 3%

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

k_min = 1; 
for t = 1:n_steps
    fprintf('t=%d (from i=%d)\n', t, i_exp)        
    % EM Algorithm    
    [GMM{t}, best_models_by_k{t}, CIC{t}, nCEs_for_best_models{t}, k_min] = EM_Algo( X, Hx_Wx, current_est, k_min ); 
    % Sampling from the last GMM
    X_new = random(GMM{t},n_per_step);hw

    % Evaluate the simulation output, H(x), and the likelihood ratio W(x).
    hw = pdf(GMM{t},X_new);
    X = [X; X_new]; Hx_new = H(X_new); Wx_new = fu(X_new)./hw; %#ok<AGROW>
    Hx = [Hx; Hx_new];  Wx = [Wx; Wx_new]; %#ok<AGROW>
    Hx_Wx = [Hx_Wx; Hx_new.*Wx_new]; %#ok<AGROW>
    disp(size(Hx_Wx));
    % Current POE estimate
    current_est = mean(Hx_Wx((n_per_initstep+1):end)); 
end

%% Final sampling and estimation
	fprintf('t=%d (from i=%d)\n', t+1, i_exp) 
    [GMM_final, best_models_by_k_final, CIC_final, nCEs_for_best_models_final, k_min] = EM_Algo( X, Hx_Wx, current_est, k_min ); 
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
end
