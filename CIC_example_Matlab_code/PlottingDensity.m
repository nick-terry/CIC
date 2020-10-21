% Draw the contour plot
load('Ex1_n_exp_500_bb1.5_kk0.1_ee0_2015-7-8-11-10.mat')

%% Optimal density plot
addpath('./mcint')
n = 1e16; %1e6; % more points = more accuracy. Relative error goes as 1/sqrt(n)
int_mcint = @(x) (H(x').*fu(x'))'; % @(x) (f(x').* s(x'))'; %Transposes are necessary to be compatible with mcint().
opt.noerror = 0;
opt.tol = 1e-7; %2e-4;
opt.time = 60; %in seconds
opt.warnOff = 0;
opt.mcintPar = [int_mcint]; %[];
h.b = [ [-Inf, Inf];[-Inf, Inf] ];
h.cond = {};
h.funcs = {@integrand_mcint}; %integrand_mcint.m  function file is created. %{(@(x) f(x').* sqrt(s_meta(x')./N_T + (1-1/N_T).*s_meta(x').^2))}; DOESN't work this way because an additional parameter needs to be passed.
mcint_output = mcint(h, n, opt);  %Note: the error is contained in mcint_output(1,1,2);  calculation code: getError(s,points,V) in mcint.m
norm_constant_IS_pdf = mcint_output(1,1,1);

f_opt = @(x,y) H([x,y]).*fu([x,y])./norm_constant_IS_pdf;

% Plot prep
x = -4:0.025:4;
y =  4:-0.025:-4;
x_len = length(x);
y_len = length(y);

x_long = repmat(x',[y_len,1]);
y_long = repelem(y',x_len);
xy_long = [x_long,y_long];
z_long = H(xy_long).*fu(xy_long)./norm_constant_IS_pdf;
z = reshape(z_long, [x_len, y_len])';


figure
colormap(gray) %bone
contourf(x,y,-z)
hold on
y_ref = (bb - kk.*(x - ee).^2);
plot(x,y_ref, 'r--')
hold off
xlabel('x_1'); ylabel('x_2')
save2pdf('Ex1_optimal_density', gcf, 600); %300); %resolution should be 300 dpi in both black and white and color cases.

%% 
i = 8;
%for i = 1:10
    GM = final_density_list_CEM{i};  
    %GM = final_density_list_KS{1};

    z_GM_long = pdf(GM,xy_long);
    z_GM = reshape(z_GM_long, [x_len, y_len])';

    figure
    colormap(gray)
    contourf(x,y,-z_GM)

    hold on
    y_ref = (bb - kk.*(x - ee).^2);
    plot(x,y_ref, 'r--')
    hold off
%end
xlabel('x_1'); ylabel('x_2')
save2pdf('Ex1_EMCE_density', gcf, 600); %300); %resolution should be 300 dpi in both black and white and color cases.

GM.NumComponents
