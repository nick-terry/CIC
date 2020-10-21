function GM_updated = M_step(X, Hx_Wx, GM, gamma)
    cov_regularization = 0; %1e-6; %'regularize' term in Figueiredo&Jain (2002) code, for numerical stability of the covariance when the sample size is small. 

    Xdim = size(X,2);     n_comp = size(gamma,2);    
    mu_old = GM.mu;
    
    Hx_Wx_gamma = bsxfun(@times, Hx_Wx, gamma);
    sum_Hx_Wx_gamma = sum(Hx_Wx_gamma,1);
    w =  sum_Hx_Wx_gamma ./ sum(Hx_Wx);

    mu = bsxfun(@rdivide, Hx_Wx_gamma' * X, sum_Hx_Wx_gamma');

    sqrt_Hx_Wx_gamma = sqrt(Hx_Wx_gamma);
    sig = zeros(Xdim, Xdim, n_comp);
    for j = 1:n_comp
        X_d = bsxfun(@minus, X, mu_old(j,:));
        X_d_sqrt_Hx_Wx_gamma = bsxfun(@times, X_d, sqrt_Hx_Wx_gamma(:,j));
        sig(:,:,j) = (X_d_sqrt_Hx_Wx_gamma'*X_d_sqrt_Hx_Wx_gamma)./sum_Hx_Wx_gamma(j) + eye(Xdim)*(cov_regularization); % cov_regularization for numerical stability;
    end       
    
    try
        GM_updated = gmdistribution(mu, sig, w);
    catch  %The mixing proportions must be positive.
        GM_updated = NaN;  %return NaN so that nCE = NaN; by the next try-catch in the EM_Algo
%         w, min(w) %#ok<NOPRT>  % NOTE: I observed that some mixing proportions are zero. 
%         error('Catched Error -- probably: The mixing proportions must be positive.')
    end
end