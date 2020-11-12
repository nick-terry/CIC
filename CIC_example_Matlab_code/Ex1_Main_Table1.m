%% Setup
ifsave = 1; %save the worskpace if 1.
n_exp = 50; %50;
randseed = 42;
rng(randseed);

%% Eq (19) and setup in p. 39 LB of KS(2013)
%%Table 1 example: bb = 5; kk = 0.5; ee = 0.1;
bb = 5; kk = 0.5; ee = 0.1; %Table 2 example (instead of bb = 5)
H = @(x) (bb - x(:,2) - kk.*(x(:,1) - ee).^2) <= 0;
fu = @(x) mvnpdf(x,zeros(1,2),eye(2));

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

% Parallel setup
n_parallel = 5; %25; 
poolobj = parpool('local',n_parallel);
spmd
s = RandStream.create('mrg32k3a','NumStreams',numlabs,'StreamIndices',labindex, 'Seed', 'shuffle');
RandStream.setGlobalStream(s);
end

%%  KS method
final_est_list_KS = nan(1,n_exp);
final_density_list_KS  = cell(1,n_exp);

tstart = tic;
parfor i = 1:n_exp
    mu0 = mu0_list{i};
    [final_est_list_KS(i), final_density_list_KS{i} ] = KSmethod( w0, mu0, sig0, H, fu );
end
telapsed = toc(tstart);


display('KS method results: ')
fprintf('Elapsed Time = %g seconds = %g minutes = %g hours\n', telapsed, telapsed/60, telapsed/(60*60))
fprintf('Sample Mean  = %g\n', mean(final_est_list_KS))
fprintf('Standard Err = %g\n', std(final_est_list_KS))
fprintf('CoV = %g\n', std(final_est_list_KS)./mean(final_est_list_KS))

%% CEM
final_est_list_CEM = nan(1,n_exp);
final_density_list_CEM  = cell(1,n_exp);
model_info  = cell(1,n_exp);

tstart_CEM = tic;
parfor i = 1:n_exp
    mu0 = mu0_list{i};
    [final_est_list_CEM(i), final_density_list_CEM{i}, model_info{i} ] = CEM( w0, mu0, sig0, H, fu );
end
telapsed_CEM = toc(tstart_CEM);


display('CEM method results: ')
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
figure
ezcontourf(@(x,y) -pdf(GM,[x y]), [-4 4], [0 5], grid_size)  %ezsurfc % ezsurf(@(x,y) -pdf(GM,[x y]), [-4 4], [0 5])
colormap(bone)%gray
hold on
bb = 2; kk = 0.1; ee = 0; %Table 2 example
x = -4:0.01:4;
y = (bb - kk.*(x - ee).^2);
plot(x,y, 'k')
hold off

% % bb =5
% ezcontourf(@(x,y) -pdf(GM,[x y]), [-6 6], [2 6], 800)
% colormap(gray)
% hold on
% bb = 5; kk = 0.1; ee = 0; %Table 2 example
% x = -6:0.01:6;
% y = (bb - kk.*(x - ee).^2);
% plot(x,y, 'k')
% hold off

%% Wrapup
delete(poolobj)
if (ifsave == 1)
    datetime = fix(clock);
    datetime_str = horzcat(num2str(datetime(1)),'-',num2str(datetime(2)),'-',num2str(datetime(3)),'-',num2str(datetime(4)),'-',num2str(datetime(5)));    
    saveFileName = horzcat('Ex1_n_exp_',num2str(n_exp),'_bb',num2str(bb),'_kk',num2str(kk),'_ee',num2str(ee),'_',datetime_str,'.mat');
    save(saveFileName);
end
