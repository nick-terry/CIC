%% bb 1.5
load('Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat')


num_components = zeros(1,n_exp);
for i = 1:n_exp
    num_components(i) = final_density_list_CEM{i}.NumComponents;
end
mean(num_components)
std(num_components)
max(num_components)
%hist(num_components)

%% bb 2.0
clear
load('Ex1_n_exp_500_bb2_kk0.1_ee0_2015-7-8-11-42.mat')


num_components = zeros(1,n_exp);
for i = 1:n_exp
    num_components(i) = final_density_list_CEM{i}.NumComponents;
end
mean(num_components)
std(num_components)
%hist(num_components)


%% bb 2.5
clear
load('Ex1_n_exp_500_bb2.5_kk0.1_ee0_2015-7-8-12-16.mat')


num_components = zeros(1,n_exp);
for i = 1:n_exp
    num_components(i) = final_density_list_CEM{i}.NumComponents;
end
mean(num_components)
std(num_components)
hist(num_components)