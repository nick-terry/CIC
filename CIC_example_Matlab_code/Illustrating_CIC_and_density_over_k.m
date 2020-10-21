% Goal: Illustrate how CIC and estimated densities vary over k 
% Created on 10/12/2018 

%% Load data for bb = 1.5 because the smaller its value is, the important
% region is wider (making the visualization more effective).
% All data for the Table of example 1:
% \verb|Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat|
% \verb|Ex1_n_exp_500_bb2_kk0.1_ee0_2015-7-8-11-42.mat|
% \verb|Ex1_n_exp_500_bb2.5_kk0.1_ee0_2015-7-8-12-16.mat|
load('Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat')

%% Plotting CIC for a chosen experiment 
% from CEM.m: model_info = {best_models_by_k_final, CIC_final, nCEs_for_best_models_final, Hx_Wx, GMM, best_models_by_k, CIC, nCEs_for_best_models};
i_exp = 3; %chosen index of an experiment
best_models_by_k_final = model_info{i_exp}{1}; %best GMM models as functions of k at the final iteration (t = \tau)
CIC_final = model_info{i_exp}{2};
CIC = model_info{i_exp}{7};

%color
r_color = [228,26,28]/255; g_color = [55,126,184]/255; b_color = [77,175,74]/255;
% Other three colors from colorbrewer2
%[127,201,127]/255)
%[190,174,212]/255
%[253,192,134]/255

figure
plot(-CIC{1},'Color', r_color, 'LineStyle','-')  %CIC of the 1st (not 0th) iteration 
hold on
plot(-CIC{4},'Color', g_color, 'LineStyle','-.')  %Instead of for-loop: n_steps = 6; for t = 1:n_steps 
plot(-CIC_final,'Color', b_color, 'LineStyle','--') 

t=1; k=1;
scatter(k,-CIC{t}(k),[],'MarkerEdgeColor', r_color)
t=1; [~,k] = min(-CIC{t}); %k that minimizes the CIC.
scatter(k,-CIC{t}(k),[],'MarkerEdgeColor', r_color)
t=1; k=10; 
scatter(k,-CIC{t}(k),[],'MarkerEdgeColor', r_color)
% t=4; k=4;  %-CIC{t}(k) = 0.162002350393683
% scatter(k,-CIC{t}(k),[],'MarkerEdgeColor', g_color)
t=4; [~,k] = min(-CIC{t}); %-CIC{t}(k) = 0.161895249863414
scatter(k,-CIC{t}(k),[],'MarkerEdgeColor', g_color)
[~,k] = min(-CIC_final);
scatter(k,-CIC_final(k),[],'MarkerEdgeColor', b_color)

hold off
legend('CIC at t = 1 (cumulated sample size = 2000)','CIC at t = 4 (cumulated sample size = 5000)','CIC at t = 7 (cumulated sample size = 8700)')
xlabel('Number of Components, k'); ylabel('CIC')

save2pdf('CIC_over_k_from_Illustrating_CIC', gcf, 600);

%% Plotting densities
% Plot prep
x = -4:0.025:4;
y =  4:-0.025:-4;
y_ref = (bb - kk.*(x - ee).^2);

best_models_by_k = model_info{i_exp}{6};

% Plotting GMMs
t = 1; k = 1;
Plot_GMM(best_models_by_k{t}{k},x,y,y_ref) %GMM for t = 1, k = 1
save2pdf(strcat('CIC-IS_density_t_',num2str(t),'_k_',num2str(k)), gcf, 600);

t = 1; [~,k] = min(-CIC{t}); 
Plot_GMM(best_models_by_k{t}{k},x,y,y_ref) %GMM for t = 1, k = 7
save2pdf(strcat('CIC-IS_density_t_',num2str(t),'_k_',num2str(k)), gcf, 600);
 
% t = 1; k = 10;
% Plot_GMM(best_models_by_k{t}{k},x,y,y_ref) %GMM for t = 1, k = 1
% save2pdf(strcat('CIC-IS_density_t_',num2str(t),'_k_',num2str(k)), gcf, 600);

t = 4; k = 4;
Plot_GMM(best_models_by_k{t}{k},x,y,y_ref) %GMM for t = 4, k = 4
save2pdf(strcat('CIC-IS_density_t_',num2str(t),'_k_',num2str(k)), gcf, 600);

t = 4; [~,k] = min(-CIC{t}); 
Plot_GMM(best_models_by_k{t}{k},x,y,y_ref) %GMM for t = 4, k = 4
save2pdf(strcat('CIC-IS_density_t_',num2str(t),'_k_',num2str(k)), gcf, 600);

k = 8; 
Plot_GMM(best_models_by_k_final{k},x,y,y_ref)%GMM for t = 7 (final iteration), k = 8
save2pdf(strcat('CIC-IS_density_t_7_k_',num2str(k)), gcf, 600);

