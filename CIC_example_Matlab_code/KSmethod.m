function [ final_est, final_density ] = KSmethod( w0, mu0, sig0, H, fu )
% Kurtz and Song (2013) method
%   w0: initial weights : n_comp by 1, where n_comp is the number of component densities.
%   mu0: initial mean vectors : n_comp by Xdim, where Xdim is the dimension of X.
%   sig0: initial covariances : n_comp by n_comp

n_comp = length(w0);
Xdim = size(mu0,2);


%% Setup used for Table 2 in KS(2013)
n_steps = 6;
n_per_step = 10^3; 
n_final = 1700; % the case corresponding to c.o.v. = 3%

%% t = 0
initGM = gmdistribution(mu0, sig0, w0);
X = random(initGM, n_per_step);

%% t = 1
mu  = cell(1,n_steps);
sig = cell(1,n_steps);
w   = cell(1,n_steps);
GMM = cell(1,n_steps);

hw = pdf(initGM,X);
gamma = posterior(initGM,X);
Hx = H(X);
Wx = fu(X)./hw;

Hx_Wx = Hx.*Wx;
Hx_Wx_gamma = bsxfun(@times, Hx_Wx, gamma);

sum_Hx_Wx_gamma = sum(Hx_Wx_gamma,1);
w{1} =  sum_Hx_Wx_gamma ./ sum(Hx_Wx);

mu{1} = bsxfun(@rdivide, Hx_Wx_gamma' * X, sum_Hx_Wx_gamma');

sqrt_Hx_Wx_gamma = sqrt(Hx_Wx_gamma);
sig{1} = zeros(Xdim, Xdim, n_comp);
for j = 1:n_comp
    X_d = bsxfun(@minus, X, mu0(j,:));
    X_d_sqrt_Hx_Wx_gamma = bsxfun(@times, X_d, sqrt_Hx_Wx_gamma(:,j));
    sig{1}(:,:,j) = (X_d_sqrt_Hx_Wx_gamma'*X_d_sqrt_Hx_Wx_gamma)./sum_Hx_Wx_gamma(j); %+eye(Xdim)*(1e-06); % add a small identity matrix for numerical stability;
end

GMM{1} = gmdistribution(mu{1}, sig{1}, w{1});

%% t = 2 through n_steps
for t = 2:n_steps
    X = random(GMM{t-1}, n_per_step);
    hw = pdf(GMM{t-1},X);
    gamma = posterior(GMM{t-1},X);
    Hx = H(X);
    Wx = fu(X)./hw;

    Hx_Wx = Hx.*Wx;
    Hx_Wx_gamma = bsxfun(@times, Hx_Wx, gamma);

    sum_Hx_Wx_gamma = sum(Hx_Wx_gamma,1);
    w{t} =  sum_Hx_Wx_gamma ./ sum(Hx_Wx);

    mu{t} = bsxfun(@rdivide, Hx_Wx_gamma' * X, sum_Hx_Wx_gamma');

    sqrt_Hx_Wx_gamma = sqrt(Hx_Wx_gamma);
    sig{t} = zeros(Xdim, Xdim, n_comp);
    for j = 1:n_comp
        X_d = bsxfun(@minus, X, mu{t-1}(j,:));
        X_d_sqrt_Hx_Wx_gamma = bsxfun(@times, X_d, sqrt_Hx_Wx_gamma(:,j));
        sig{t}(:,:,j) = (X_d_sqrt_Hx_Wx_gamma'*X_d_sqrt_Hx_Wx_gamma)./sum_Hx_Wx_gamma(j);
    end    
    
    GMM{t} = gmdistribution(mu{t}, sig{t}, w{t});
end

final_density = GMM{n_steps};
X = random(final_density, n_final);
hw = pdf(final_density,X);
Hx = H(X);
Wx = fu(X)./hw;
final_est = mean(Hx.*Wx);


end


