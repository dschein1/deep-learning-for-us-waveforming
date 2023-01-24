function display_comparision(method, num_cycles, depth)
    success_name = 'py to matlab/success rate 3 repeats 10,000 each gs,net single,net multi ';
    MSE_conv_image = 'results/results comparision success MSE conv only gs 0 ';
    success_rate_image = 'results/results comparision success only gs 0 ';
    MSE_image = 'results/results comparision success MSE only gs 0 ';
    if strcmp(method,'integers')
        success_name = [success_name 'integers '];
        success_rate_image = [success_rate_image 'integers '];
        MSE_image = [MSE_image 'integers '];
        MSE_conv_image = [MSE_conv_image 'integers '];
    end
    
    success_name = [success_name int2str(depth) int2str(num_cycles) ' cycles.mat'];
    success_rate_image = [success_rate_image int2str(num_cycles) ' cycles.tif'];
    MSE_image = [MSE_image int2str(num_cycles) ' cycles.tif'];
    MSE_conv_image = [MSE_conv_image int2str(num_cycles) ' cycles.tif'];
    full = load(success_name);
    total_results = full.full.total_results;
    total_amounts = full.full.total_amount;
    pct = zeros(2,10,3);
    pct(1,:,:)= squeeze(total_results(1,:,:)) ./ total_amounts;
    pct(2,:,:) = squeeze(total_results(2,:,:)) ./ total_amounts;
    %pct(3,:,:) = squeeze(total_results(3,:,:)) ./ total_amounts;
    means = mean(pct,3)';
    stds = std(pct,0,3)';
    
    y = squeeze(means);
    error_bar = squeeze(stds);
    
    fig = figure;
    h = plot(y,'LineWidth',2); axis tight;
    ax = gca;
    ax.FontSize = 26;
    ylabel('Success rate','FontSize',26); xlabel('Number of foci','FontSize',26)
    legend({'GS 1-cycle' 'USDL'},'FontSize',26,'Location','northeast')
    hold on
    errorbar(y,error_bar,'k','linestyle','none','HandleVisibility','off','LineWidth',1);
    % 
    grid off
    hgexport(gcf, [success_rate_image], hgexport('factorystyle'), 'Format', 'tiff');
    

% grid on
    % hgexport(gcf, [MSE_image], hgexport('factorystyle'), 'Format', 'tiff');
    total_results = full.full.total_mse_conv;
    pct = zeros(2,10,3);
    pct(1,:,:)= squeeze(total_results(1,:,:)) ./ total_amounts;
    pct(2,:,:) = squeeze(total_results(2,:,:)) ./ total_amounts;
    %pct(3,:,:) = squeeze(total_results(3,:,:)) ./ total_amounts;
    means = mean(pct,3)';
    stds = std(pct,0,3)';
    
    y = squeeze(means);
    error_bar = squeeze(stds);
    figure
    h = plot(y,'LineWidth',2); axis tight; 
    ax = gca;
    ax.FontSize = 26;
    ylabel('MSE','FontSize',26); xlabel('Number of foci','FontSize',26)
    legend({'GS 1-cycle' 'USDL' },'FontSize',26,'Location','northwest')
    hold on
    errorbar(y,error_bar,'k','linestyle','none','HandleVisibility','off','LineWidth',1);
    % 
    grid off
    hgexport(gcf, [MSE_conv_image], hgexport('factorystyle'), 'Format', 'tiff');
end