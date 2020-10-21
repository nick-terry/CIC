function [gamma] = E_step(X,GM)        
    gamma = posterior(GM,X);  
end