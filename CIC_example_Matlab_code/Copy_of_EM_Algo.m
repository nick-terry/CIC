function [best_model, best_models_by_k, CIC_for_best_models, nCEs_for_best_models, k_min_updated] = EM_Algo( X, Hx_Wx, current_est, k_min )
%7/4/2015: Called by CEM.m
    [n_X,Xdim] = size(X);
    n_init = 10;     
    max_free_param = floor(n_X ./ 10); %maximum number of free parameters we would like to allow.  If the divisor is 10, it means we expect that each component has 10 observations on average. 
    k_max = floor((max_free_param+1)./(Xdim + (Xdim*(Xdim+1))./2 + 1)); % based on "max_free_param = (K-1) + K(p + p(p+1)/2)"
    k_max_reached = false;
    
    c_spread = 3;
    
    CIC_MA_window = 4;
    
    init_fail_proportion = 0.5; %If more than this proportion of the random initialization fail, k_max is reached.
    singularity_threshold = 1e5; %https://en.wikipedia.org/wiki/Condition_number  %http://mathworld.wolfram.com/ConditionNumber.html    
    tol_converg = 1e-5;
    maxiter = 500;  
    
    best_models_by_k = cell(1,k_max); %Best model for each k
    nCEs_for_best_models = nan(1,k_max);
    CIC_for_best_models = nan(1,k_max);
    CIC_MA = nan(1,k_max);
    
    
    k = k_min;
    while ~k_max_reached && k <= k_max       
%          fprintf('    k=%d\n', k) 
        GM_list =  cell(1,n_init);
        nCE_list = nan(1,n_init);
        for i_init = 1:n_init
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
            
            iter = 1; nCE_old = -Inf; converged = false; nCE = NaN; %negative cross-entropy estimate
            while ~converged && iter <= maxiter      
%                 if mod(iter,100)==0 
%                     fprintf('        Iteration (i_init=%d): %d / %d\n', i_init, iter, maxiter) 
%                 end
                gamma = E_step(X,GM);
                GM =  M_step(X, Hx_Wx, GM, gamma); %Left-hand side GM is the updated model.
                if ~isa(GM,'gmdistribution'), nCE = NaN; break; end %Some mixing proportions are zero.
                 
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
        
        if k >= (k_min + CIC_MA_window - 1)
            CIC_MA(k) = nanmean(CIC_for_best_models((k-CIC_MA_window+1):k));
            if k > (k_min + CIC_MA_window - 1)
                if(CIC_MA(k)-CIC_MA(k-1) < 0), k_max_reached = true; break; end
            end
        end
        
        n_failed_init = sum(isnan(nCE_list));
        k_max_reached =  n_failed_init > n_init.*init_fail_proportion; 
        k = k + 1;
        if((k-1) == k_min && n_failed_init == n_init) %if all initializations are failed at k_min, do k-step again.
            k_max_reached = false;
            k_min = max(1, k_min -1);
            k = k_min;
            fprintf('    k=%d (restart)\n', k) 
        end
    end
%     best_models_by_k = best_models_by_k(k_min:(k-1));
%     nCEs_for_best_models = nCEs_for_best_models(k_min:(k-1));
%     CIC_for_best_models = CIC_for_best_models(k_min:(k-1)); 
    
    %k_list = k_min:length(best_models_by_k);
    %d_list = k_list - 1 + k_list.*(Xdim + Xdim*(Xdim+1)/2); %number of free parameters
    %CIC = nCEs_for_best_models - d_list./n_X;
    
    [~,idx] = max(CIC_for_best_models);
    best_model = best_models_by_k{idx};
    k_min_updated = max(1, idx-3);
end

%% [gamma] = E_step(X,GM)           

%% GM_updated = M_step(X, Hx_Wx, GM, gamma)

