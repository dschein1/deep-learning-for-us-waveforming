function save_example_to_file(patterns, delays_gs, delays_0, file_path)
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3;
    pitch_half = pitch/2;
    dz = 0.1e-3;
    z = 20e-3:dz:60e-3;
    x_min = -17e-3;
    x_max = 17e-3;
    z_min = 20e-3;
    z_max = 60e-3;
    N = 1024;
    range_x = round(x_max / pitch_half);
    x_for_image = (-range_x * pitch_half):pitch_half:range_x*pitch_half;
    x = linspace(x_min,x_max,200);
    z_field = linspace(z_min,z_max,300);
    im_gs = squeeze(create_new_image(delays_gs));
    im_0 = squeeze (create_new_image(delays_0));

    [amps, new_transducer] = calculateGS(patterns,true);
    new_transducer = amps .* exp(1i * new_transducer);

    new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
    new_transducer = new_transducer(1,256:768 -1);
    new_transducer = imresize(new_transducer,2,'box');

    new_transducer = new_transducer(1,1:N);

    clear E_FSP1_z0
    index = 1;
    for z0 = z
        [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
        index = index + 1;
        z0/z_max;
    end
    I_FSP1_z0 = abs(E_FSP1_z0).^2;
    I_FSP1_z0 = I_FSP1_z0 ./ max(max(I_FSP1_z0));

    I_FSP1_z0 = I_FSP1_z0(:,512 - range_x:512 + range_x);
    line_gs = I_FSP1_z0(round(size(I_FSP1_z0,1)/2),:) - min(I_FSP1_z0(round(size(I_FSP1_z0,1)/2),:));
    line_at_depth_gs_im = im_gs(round(size(im_gs,1)/2),:)' - min(im_gs(round(size(im_gs,1)/2),:))';
    line_at_depth_gs_im = line_at_depth_gs_im ./ max(line_at_depth_gs_im);
    line_at_depth_0_im = im_0(round(size(im_0,1)/2),:)' - min(im_0(round(size(im_0,1)/2),:))';
    line_at_depth_0_im = line_at_depth_0_im ./ max(line_at_depth_0_im);
    
    bottom = min([min(min(I_FSP1_z0)),min(min(im_gs)),min(min(im_0))]);
    top  = max([max(max(I_FSP1_z0)),max(max(im_gs)),max(max(im_0))]);
    fig = figure;
    fig;
    fig.Position = [100 100 1280 * 1 1020 * 1];
    ylabel('Intensity a.u')
    a = subplot(2,2,1);
    imagesc(x_for_image * 1e3,z * 1e3,I_FSP1_z0); colormap hot; axis square; axis on; zoom(1);
    a.FontSize = 26;
    xlabel('x [mm]','FontSize',26); ylabel('z [mm]', FontSize=26);
    pos1 = get(a,'Position');
    caxis manual
    caxis([bottom top]);
    colorbar('eastoutside')
    set(a,'Position',pos1)
    a = subplot(2,2,2);
    imagesc(x * 1e3,z_field * 1e3,im_gs);colormap hot; axis square; axis on; zoom(1);
    set(gca,'xticklabel',[]) %                xlabel('x [mm]','FontSize',18); ylabel('z [mm]', FontSize=18);
    set(gca,'yticklabel',[])
    caxis manual
    caxis([bottom top]);
    a = subplot(2,2,3);
    imagesc(x * 1e3,z_field * 1e3,im_0);colormap hot; axis square; axis on; zoom(1);
    set(gca,'xticklabel',[]) 
    set(gca,'yticklabel',[])
    caxis manual
    caxis([bottom top]);   
    a = subplot(2,2,4);
    plot(x * 1e3,line_at_depth_gs_im,LineWidth=2); axis tight square
    hold on
    plot(x * 1e3,line_at_depth_0_im,LineWidth=2);
    a.FontSize = 26;
    xlabel('x [mm]','FontSize',26); ylabel('Intensity [a.u]', FontSize=26);
    style = hgexport('factorystyle');
    style.Bounds = 'loose';
    hgexport(gcf, [file_path], style, 'Format', 'tiff');
    close all