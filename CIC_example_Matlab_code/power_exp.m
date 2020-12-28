%% Setup
ifsave = 1; %save the worskpace if 1.
n_exp = 28; %50;
randseed = 100;
rng(randseed);

%% Load saved sim results for power grid
hdict = py.importlib.import_module('hdict');
nComponents = 5; % the number of components being simulated in power grid
H = @(x) double(hdict.h5(x));
fu = @(x) mvnpdf(x,zeros(1,nComponents),eye(nComponents));

%% Initial distribution
n_comp = 10;
Xdim = nComponents;
c_spread = 3;

mu0_list = cell(1,n_exp);
for i = 1:n_exp
    mu0_list{i} = mvnrnd(zeros(n_comp,Xdim),eye(Xdim));
end

%mu0 = mvnrnd(zeros(n_comp,Xdim),eye(Xdim));
sig0 = c_spread*eye(Xdim);
w0 = ones(1,n_comp)./n_comp;

% Parallel setup
% n_parallel = 14; %25; 
% poolobj = parpool('local',n_parallel);
% spmd
% s = RandStream.create('mrg32k3a','NumStreams',numlabs,'StreamIndices',labindex, 'Seed', 'shuffle');
% RandStream.setGlobalStream(s);
% end

%%  KS method
% final_est_list_KS = nan(1,n_exp);
% final_density_list_KS  = cell(1,n_exp);
% 
% tstart = tic;
% parfor i = 1:n_exp
%     mu0 = mu0_list{i};
%     [final_est_list_KS(i), final_density_list_KS{i} ] = KSmethod( w0, mu0, sig0, H, fu );
% end
% telapsed = toc(tstart);
% 
% 
% display('KS method results: ')
% fprintf('Elapsed Time = %g seconds = %g minutes = %g hours\n', telapsed, telapsed/60, telapsed/(60*60))
% fprintf('Sample Mean  = %g\n', mean(final_est_list_KS))
% fprintf('Standard Err = %g\n', std(final_est_list_KS))
% fprintf('CoV = %g\n', std(final_est_list_KS)./mean(final_est_list_KS))

%% CEM
final_est_list_CEM = nan(1,n_exp);
final_density_list_CEM  = cell(1,n_exp);
model_info  = cell(1,n_exp);

tstart_CEM = tic;
for i = 1:n_exp
    mu0 = mu0_list{i};
    [final_est_list_CEM(i), final_density_list_CEM{i}, model_info{i} ] = CEM( w0, mu0, sig0, H, fu, i );
end
telapsed_CEM = toc(tstart_CEM);


disp('CEM method results: ')
fprintf('Elapsed Time = %g seconds = %g minutes = %g hours\n', telapsed_CEM, telapsed_CEM/60, telapsed_CEM/(60*60))
fprintf('Sample Mean  = %g\n', mean(final_est_list_CEM))
fprintf('Standard Err = %g\n', std(final_est_list_CEM))
fprintf('CoV = %g\n', std(final_est_list_CEM)./mean(final_est_list_CEM))

%% Plotting

% bb =2
GM = final_density_list_CEM{1};   %8
%GM = model_info{1,8}{1,1}{1,12};
%GM = final_density_list_KS{1};
grid_size = 200; %60; %200;

% figure
% ezcontourf(@(x,y) -pdf(GM,[x y]), [-6 6], [2 6], grid_size) %ezsurfc % ezsurf(@(x,y) -pdf(GM,[x y]), [-4 4], [0 5]  [-6 6], [2 6]
% colormap(bone) %gray
% hold on
% x = -6:0.01:6;
% y = (bb - kk.*(x - ee).^2);
% plot(x,y, 'r')
% hold off

%% Wrapup
% delete(poolobj)
if (ifsave == 1)
    datetime = fix(clock);
    datetime_str = horzcat(num2str(datetime(1)),'-',num2str(datetime(2)),'-',num2str(datetime(3)),'-',num2str(datetime(4)),'-',num2str(datetime(5)));    
    saveFileName = horzcat('Ex1_n_exp_powergrid_10_',datetime_str,'.mat');
    save(saveFileName);
end
