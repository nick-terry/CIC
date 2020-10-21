function integrand = integrand_mcint(x, par)
global p_var;
global N_T;
global mu_hat;
global sigma_hat;
global target_quantile; %for mcint

x_mu = zeros(1,p_var); %input variable
x_sigma = eye(p_var);  %input variable

f = (@(x) mvnpdf(x,x_mu,x_sigma));   %f = (@(x) mvnpdf(x,x_mu(1:size(x,2)),x_sigma(1:size(x,2), 1:size(x,2)))); %Variation for 3d integration        
s_meta = (@(x) 1-normcdf(target_quantile, mu_hat(x), sigma_hat(x)));   


integrand = (f(x').* sqrt(s_meta(x')./N_T + (1-1/N_T).*s_meta(x').^2))';%  f(x).* sqrt(s_meta(x)./N_T + (1-1/N_T).*s_meta(x).^2);
 

end


