function Plot_GMM(GM,x,y,y_ref)
%PLOT_GMM Draw a GMM countour plot
%   Called by Illustrating_CIC_and_density_over_k.m
    x_len = length(x);
    y_len = length(y);

    x_long = repmat(x',[y_len,1]);
    y_long = repelem(y',x_len);
    xy_long = [x_long,y_long];

    z_GM_long = pdf(GM,xy_long);
    z_GM = reshape(z_GM_long, [x_len, y_len])';

    figure
    colormap(gray)
    contourf(x,y,-z_GM)

    hold on    
    plot(x,y_ref, 'r--')
    hold off

    xlabel('x_1'); ylabel('x_2')
end

