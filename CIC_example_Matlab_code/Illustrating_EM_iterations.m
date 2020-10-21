% Goal: Illustrate how the EM iterations for CE minimization work
% Created on 10/17/2018 

% Reference code: U:\CIC\Illustrating_EM_converged.m

%% K = 3 using EM algorithm 
clear
K_fix = 3;
%% Setup
n_exp = 1;
randseed = 1000;
rng(randseed);

idx_chosen_init = 9; %Output of '[~,idx] = max(CIC_for_best_models);' -- it depends on randseed. It is chosen because we'll present the converged GMM ultimately.


%% Eq (19) and setup in p. 39 LB of KS(2013)
%%Table 1 example: bb = 5; kk = 0.5; ee = 0.1;
bb = 1.5; kk = 0.1; ee = 0; %Table 2 example (instead of bb = 5)
H = @(x) (bb - x(:,2) - kk.*(x(:,1) - ee).^2) <= 0;
fu = @(x) mvnpdf(x,zeros(1,2),eye(2));


%% Prep plotting
half_display_range = 4;
x = -half_display_range:0.025:half_display_range;
y =  half_display_range:-0.025:-half_display_range;
y_ref = (bb - kk.*(x - ee).^2);
bool_savePDF = true; %false;

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

%% t = 1 
% Reference scripts:    U:\CIC\Illustrating_EM_converged.m
%                       U:\CIC\EM_Algo_fixed_K.m                    
k_min = K_fix; k_max = K_fix;

    [n_X,Xdim] = size(X);
    n_init = 10;     
    max_free_param = floor(n_X ./ 10); %maximum number of free parameters we would like to allow.  If the divisor is 10, it means we expect that each component has 10 observations on average. 
    %k_max = floor((max_free_param+1)./(Xdim + (Xdim*(Xdim+1))./2 + 1)); % based on "max_free_param = (K-1) + K(p + p(p+1)/2)"
    k_max_reached = false;
    
    c_spread = 3;
    
    CIC_MA_window = 4;
    
    init_fail_proportion = 0.5; %If more than this proportion of the random initialization fail, k_max is reached.
    singularity_threshold = 1e5; %https://en.wikipedia.org/wiki/Condition_number  %http://mathworld.wolfram.com/ConditionNumber.html    
    tol_converg = 1e-2;%1e-5;
    maxiter = 10; %500;  
    
    best_models_by_k = cell(1,k_max); %Best model for each k
    nCEs_for_best_models = nan(1,k_max);
    CIC_for_best_models = nan(1,k_max);
    CIC_MA = nan(1,k_max);
    
    
    k = k_min;
%     while ~k_max_reached && k <= k_max       
%          fprintf('    k=%d\n', k) 
        GM_list =  cell(1,n_init);
        nCE_list = nan(1,n_init);
        for i_init = 1:n_init %I'm keeping the for-loop for the reproducibility of random data with the current 'randseed'.
            ind_positive_H = Hx_Wx > 0;
            n_positive_H = sum(ind_positive_H);
            if k >  n_positive_H
                mu = X(ind_positive_H,:);
                ind_zero_H = find(~ind_positive_H);
                mu = [mu; X(randsample(ind_zero_H,k-n_positive_H),:)];
            else
                mu = X(ind_positive_H,:);
                mu = mu(randsample(n_positive_H,k),:);
            end
            %mu = X(randsample(n_X,k),:);
            sig = c_spread.*trace(cov(X))./Xdim.*eye(Xdim); % normalized by the number of observations, n-1.
            w = ones(1,k)./k;
            GM = gmdistribution(mu, sig, w);            
            if i_init == idx_chosen_init
                %break; %For testing different plots of iter=0.
                iter = 0;
                Plot_GMM_contour_with_points(GM,K_fix,iter,X,x,y,y_ref,half_display_range,bool_savePDF)
            end            
            
            iter = 1; nCE_old = -Inf; converged = false; nCE = NaN; %negative cross-entropy estimate
            while ~converged && iter <= maxiter      
%                 if mod(iter,100)==0 
%                     fprintf('        Iteration (i_init=%d): %d / %d\n', i_init, iter, maxiter) 
%                 end
                gamma = E_step(X,GM);
                GM =  M_step(X, Hx_Wx, GM, gamma); %Left-hand side GM is the updated model.
                if ~isa(GM,'gmdistribution'), nCE = NaN; break; end %Some mixing proportions are zero.                
                
                if i_init == idx_chosen_init
                    Plot_GMM_contour_with_points(GM,K_fix,iter,X,x,y,y_ref,half_display_range,bool_savePDF)
                end
                
                % Checking the singularity of the convariance matrices (note: checking the smallest component is not enough as they also can have a well-conditioned cov.) %[~,idx] = min(GM.PComponents);
                singular_bool = false;
                for idx = 1:k           
                    if (cond(GM.Sigma(:,:,idx)) > singularity_threshold)
                       singular_bool = true;
                       break
                    end   
                end
                
                if (~singular_bool)
                    nCE = negCE(X, Hx_Wx, GM);       %nCE = mean(Hx_Wx.*log(pdf(GM,X))); %numerical errors lead to NaN.             
                else
                    nCE = NaN;  
                end
                if isnan(nCE), break; end %At least one ill-conditioned covariance is observed.                
               

                % Checking the convergence
                converged = nCE-nCE_old < tol_converg*abs(nCE_old);                
                iter = iter+1; nCE_old = nCE;
            end
            GM_list{i_init} = GM;
            nCE_list(i_init) = nCE;
        end     
        
        [~,idx] = max(nCE_list);
        best_models_by_k{k} = GM_list{idx};
        nCEs_for_best_models(k) = nCE_list(idx);
        CIC_for_best_models(k) = nCE_list(idx) - (k - 1 + k.*(Xdim + Xdim*(Xdim+1)/2))./n_X .*current_est;
        
%     end

%% Plotting for the converged GMM
% % Plot prep
% half_display_range = 4;
% x = -half_display_range:0.025:half_display_range;
% y =  half_display_range:-0.025:-half_display_range;
% x_len = length(x);
% y_len = length(y);
% 
% x_long = repmat(x',[y_len,1]);
% y_long = repelem(y',x_len);
% xy_long = [x_long,y_long];
% 
% t=1;
% GM = best_models_by_k{k};  
% %GM = GMM{1}; %First Step: n = 2000
%  z_GM_long = pdf(GM,xy_long);
%     z_GM = reshape(z_GM_long, [x_len, y_len])';
% 
%     figure
%      colormap(gray)
%      contourf(x,y,-z_GM)
% 
%     hold on
%     y_ref = (bb - kk.*(x - ee).^2);
%     plot(x,y_ref, 'r--')
%     hold off
% 
% xlabel('x_1'); ylabel('x_2')
% hold on
% xlim([-half_display_range, half_display_range])
% ylim([-half_display_range, half_display_range])
% 
% scatter(X(:,1),X(:,2),2,'MarkerEdgeColor',[218,165,32]/255,'MarkerFaceColor',[218,165,32]/255)%https://www.rapidtables.com/web/color/RGB_Color.html
% 
% %ezcontour(@(x,y)pdf(GM,[x y]),[-half_display_range half_display_range],[-half_display_range half_display_range])
% 
% MVN_pdf=cell(1,K_fix);
% for k =1:K_fix
%     MVN_pdf{k} = @(x,y)mvnpdf([x y],GM.mu(k,:),GM.Sigma(:,:,k));    
%     %Plot of proportion-weighted densities
%     fcontour(@(x,y) GM.ComponentProportion(k).*MVN_pdf{k}(x,y),[-half_display_range half_display_range -half_display_range half_display_range])   
% end
% hold off
% save2pdf('EM_iteration_coverged_2', gcf, 600);

