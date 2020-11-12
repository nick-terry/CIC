function llh = logdensity(X, GM)
%LOGMVNPDF Summary of this function goes here
%   Detailed explanation goes here
    n_X = size(X,1);  n_comp = GM.NComponents;
    logmvnpdf = zeros(n_X,n_comp);
    for k = 1:n_comp
        logmvnpdf(:,k) = log_mvnpdf(X,GM.mu(k,:),GM.Sigma(:,:,k));
    end
    
    if sum(isnan(logmvnpdf(1,:)))~=0, llh = NaN;  %If any covariance is not PD, then return NaN.
    else
        log_pdf_proportion = bsxfun(@plus,logmvnpdf,log(GM.PComponents));  %log (mvn pdf for k * weight for k)   %GM.PComponents in 2014a ; GM.ComponentProportion in matlab 2015a
        llh = LogSumExp(log_pdf_proportion,2);
    end
end

function logmvnpdf = log_mvnpdf(X, mu, sig) 
%Reference codes:
%http://www.mathworks.com/matlabcentral/fileexchange/34064-log-multivariate-normal-distribution-function/content/logmvnpdf.m
%loggausspdf in emgm package
    Xdim = size(X,2);
    X_d = bsxfun(@minus,X,mu);
    [U,p]= chol(sig);
    if p ~= 0 %if sig is not positive definite
        logmvnpdf = nan(size(X,1),1);    
    else
        sqrt_mahal = U'\X_d'; %mldivide, \
        mahal = dot(sqrt_mahal,sqrt_mahal,1)';  %Mahalanobis distance:  see the function 'mahal'
        log_mvn_const = Xdim*log(2*pi) + 2*sum(log(diag(U)));   % log (mvn constant)
        logmvnpdf = -(log_mvn_const+mahal)./2;
    end

end

function llh = LogSumExp(x, dim)
% Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
% Reference code: logsumexp in emgm package
% x : n by k matrix
% x = log(w_k * p_k), where w_k is the weight for component k, p_k is the density for component k

% subtract the largest in each column
y = max(x,[],dim);   
x = bsxfun(@minus,x,y);         %log(w_k * p_k) - y
llh = y + log(sum(exp(x),dim));   %y +  log ( sum_k ( w_k p_k *exp(-y) ) ) = y + log(exp(-y)) + log(sum_k(w_k p_k))

end