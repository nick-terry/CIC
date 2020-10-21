function Plot_GMM_contour_with_points(GM,K_fix,iter,X,x,y,y_ref,half_display_range,bool_savePDF)
%Plot_GMM_contour_with_points draws contour plots on top of sampled points.
%   Called by Illustrating_EM_iterations.m
%   Generate warnings which can be ignored.

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
    %y_ref = (bb - kk.*(x - ee).^2);
    plot(x,y_ref, 'r--')
    hold off

xlabel('x_1'); ylabel('x_2')
hold on
xlim([-half_display_range, half_display_range])
ylim([-half_display_range, half_display_range])

scatter(X(:,1),X(:,2),2,'MarkerEdgeColor',[218,165,32]/255,'MarkerFaceColor',[218,165,32]/255)%https://www.rapidtables.com/web/color/RGB_Color.html

%ezcontour(@(x,y)pdf(GM,[x y]),[-half_display_range half_display_range],[-half_display_range half_display_range])

MVN_pdf=cell(1,K_fix);
for k =1:K_fix
    if GM.SharedCovariance==false
        GM_covariance = GM.Sigma(:,:,k);
    else
        GM_covariance = GM.Sigma(:,:,1);
    end    
    MVN_pdf{k} = @(x1,x2)mvnpdf([x1,x2],GM.mu(k,:),GM_covariance);    
    %Plot of proportion-weighted densities
    fcontour(@(x1,x2) GM.ComponentProportion(k).*MVN_pdf{k}(x1,x2),[-half_display_range half_display_range -half_display_range half_display_range])   
end
hold off

if bool_savePDF == true
    save2pdf(strcat('EM_iteration_',num2str(iter)), gcf, 600);    
end

end

