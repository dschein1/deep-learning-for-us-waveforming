init_field
%%
pitch = 0.218e-3;
delays_calc = calc_delay(128,pitch,1490,[0 0 40]/1000); 
pattern= zeros(1,1024);
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
loaded_base = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\single_data0.mat');
base_delay = normalize_delays(loaded_base.from_net);
loaded_1 = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\single_data1.mat');
delay_1 = normalize_delays(loaded_1.from_net);
image_base = squeeze(create_new_image(base_delay));
image_1 = squeeze(create_new_image(delay_1));

delay_1 - base_delay

x_min = -30e-3;
x_max = 30e-3;
z_min = 10e-3;
z_max = 80e-3;
x = linspace(x_min,x_max,200);
z_field = linspace(z_min,z_max,300);
figure
subplot(1,2,1)
imagesc(x * 1e3,z_field * 1e3,image_base); colormap jet; axis square; axis on; zoom(1); title('Image base net')
subplot(1,2,2)
imagesc(x * 1e3,z_field * 1e3,image_1); colormap jet; axis square; axis on; zoom(1); title('Image step 1 net')

line_base = squeeze(create_new_line(base_delay));
line_1 = squeeze(create_new_line(delay_1));


N = 1024;
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full = -(512-1)*pitch/2:pitch:512*pitch/2;

figure
subplot(1,2,1)
plot(x_full,line_base); title('line at 40 mm from base net')
subplot(1,3,2)
plot(x_full,line_1); title('line at 40 mm from step 1 net')

%%
figure
subplot(1,2,1)
findpeaks(line_base,x_full,'MinPeakProminence',0.05,MinPeakHeight=0.2);
[pks,locs,widths_base,proms] = findpeaks(line_base,x_full,'MinPeakProminence',0.3); title('peaks from net base')
widths_base
subplot(1,2,2)
findpeaks(line_1,x_full,'MinPeakProminence',0.05,MinPeakHeight=0.3)
[pks,locs,widths_1,proms] = findpeaks(line_1,x_full,'MinPeakProminence',0.3); title('peaks from net step 1')
widths_1

mean(widths_base - widths_1)
%%
success_gs_total = [];
success_gs_multi_total = [];
success_1_total = [];
pitch = 0.218e-3;
x_min = -(256 - 1) * pitch/3;
x_max = 256 * pitch/3;
x_full = x_min:pitch/3:x_max;
x_min = -17e-3;
x_max = 17e-3;
z_min = 20e-3;
z_max = 60e-3;
x = linspace(x_min,x_max,200);
z_field = linspace(z_min,z_max,300);
N = 1024;
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
range_x = round(17e-3 / pitch_half);
x_for_image = (-range_x * pitch_half):pitch_half:range_x*pitch_half;
dz = 0.1e-3;
z_max_gs = 2*40e-3;
z = 20e-3:dz:60e-3;
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
mode = 'floats';
if strcmp(mode,'floats')
    loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision with base model.mat');
else
    loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision with base model integers.mat');
end
db_val = 20;
const_b = 10^(-db_val/20);
const_a = 1-const_b;
delays_base = loaded_for_comparision.from_gs;
source = loaded_for_comparision.source;
x_conv = loaded_for_comparision.x_acc;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
total_size = length(delays_step_1);
size_of_step = floor(total_size / 3);
peak_height = 0.5;
total_results = zeros(3,10,3);
total_amounts = zeros(10,3);
total_mse = zeros(3,10,3);
total_mse_conv = zeros(3,10,3);
for i=1:3
    success_gs = zeros(1,10)';
    mse_gs = zeros(1,10)';
    mse_gs_conv = zeros(1,10)';
    success_0 = zeros(1,10)';
    mse_0 = zeros(1,10)';
    mse_1 = zeros(1,10)';
    mse_0_conv = zeros(1,10)';
    mse_1_conv = zeros(1,10)';
    success_1 = zeros(1,10)';
    num_for_each = zeros(10,1);
    for j=(i-1) * size_of_step + 1:i *size_of_step
        if j==46
            j;
        base_pattern = source(j,:);
        patterns = padarray(base_pattern,[0 256],0,'both');
        [amps, delays_gs] = calculateGS(patterns,false);
        delays_gs = normalize_delays(delays_gs);
        delays_0 = normalize_delays(delays_step_0(j,:));
        delays_1 = normalize_delays(delays_step_1(j,:));
        line_at_depth_gs = create_new_line(delays_gs,ones(1,128),2,3,40e-3);
        line_at_depth_0 = create_new_line(delays_0,ones(1,128),2,3,40e-3);
        %line_at_depth_1 = create_new_line(delays_1);
        [pks_base,locs_base,widths_base ,~] = findpeaks(base_pattern,x_full_gs,'MinPeakHeight',peak_height);
        locs_base = locs_base + abs(x_min);
        locs_base = round(locs_base / (pitch/3));
        pattern_for_comparision = zeros(1,512);
        pattern_for_comparision(locs_base) = pks_base;
        pattern_for_comparision_conv = interp1(x_full_gs,x_conv(j,:),x_full);
        if strcmp(mode,'floats')
            [pks_gs,locs_gs] = extract_peaks(line_at_depth_gs,pattern_for_comparision);
            [pks_0,locs_0] = extract_peaks(line_at_depth_0, pattern_for_comparision);
            %[pks_1,locs_1] = extract_peaks(line_at_depth_1,pattern_for_comparision);
        else
            [pks_gs,locs_gs,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
            [pks_0,locs_0,widths_base_0,~] = findpeaks(line_at_depth_0,x_full,'MinPeakHeight',peak_height);
            %[pks_1,locs_1,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);        
        end
        
        
        num_peaks = length(pks_base);
        if num_peaks == 3
            j;
        end
        if num_peaks < 11 && num_peaks > 0
            if (num_peaks ~= length(pks_gs) || num_peaks ~= length(pks_0))  && j > 10000000
                im_gs = squeeze(create_new_image(delays_gs));
                im_0 = squeeze (create_new_image(delays_0));
                %im_1 = squeeze(create_new_image(delays_1));

                [amps, new_transducer] = calculateGS(patterns,true);
                new_transducer = amps .* exp(1i * new_transducer);

                new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
                new_transducer = new_transducer(1,256:768 -1);
                new_transducer = imresize(new_transducer,2,'box');

                new_transducer = new_transducer(1,1:N);

                clear E_FSP1_z0
                index = 1;
                for z0 = z
                %     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
                    %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                
                    [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                    index = index + 1;
                    z0/z_max;
                end
                I_FSP1_z0 = abs(E_FSP1_z0).^2;
                I_FSP1_z0 = I_FSP1_z0 ./ max(max(I_FSP1_z0));

%                 I_FSP1_z0 = 20*log10(const_a * I_FSP1_z0 +const_b);
                
%                 line_gs = abs(FSP_X_near(new_transducer,40e-3,N,pitch_half,Wavelength)).^2;
% 
%                 %line_gs = line_gs(N/2 - 200:N/2 + 200);
%                 line_gs = interp1(x_half,line_gs,x_full);
%                 line_gs = line_gs ./ max(max(line_gs));
                I_FSP1_z0 = I_FSP1_z0(:,512 - range_x:512 + range_x);
                line_gs = I_FSP1_z0(round(size(I_FSP1_z0,1)/2),:) - min(I_FSP1_z0(round(size(I_FSP1_z0,1)/2),:));
%                 [pks_gs_base,locs_gs_base] = extract_peaks(line_gs,pattern_for_comparision);
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
%                 pos1 = get(a,'Position');
%                 colorbar('eastoutside')
%                 set(a,'Position',pos1)
                a = subplot(2,2,3);
                imagesc(x * 1e3,z_field * 1e3,im_0);colormap hot; axis square; axis on; zoom(1);
                %a.FontSize = 14;
                set(gca,'xticklabel',[]) % xlabel('x [mm]','FontSize',18); ylabel('z [mm]', FontSize=18);
                set(gca,'yticklabel',[])
                caxis manual
                caxis([bottom top]);
%                 pos1 = get(a,'Position');
%                 colorbar('eastoutside')
%                 set(a,'Position',pos1)
%   
                a = subplot(2,2,4);
                plot(x * 1e3,line_at_depth_gs_im,LineWidth=2); axis tight square
                hold on
                plot(x * 1e3,line_at_depth_0_im,LineWidth=2);
                a.FontSize = 26;
                %legend({'GS 1-cycle','USDL'},'FontSize',20,'Location','best')
                xlabel('x [mm]','FontSize',26); ylabel('Intensity [a.u]', FontSize=26);
%                 plot(x_for_image * 1e3,line_gs)%,pks_gs_base,locs_gs_base);
%                 a.FontSize = 14;
%                 xlabel('x [mm]',FontSize=14);ylabel('Intensity [a.u]','FontSize',14); axis tight
%                 %plot(x_full,pattern_for_comparision_conv);
%                 a = subplot(2,3,5);
%                 plot(x * 1e3,line_at_depth_gs_im); axis tight%,pks_gs,locs_gs);
%                 a.FontSize = 14;
%                 a = subplot(2,3,6);
%                 plot(x * 1e3,line_at_depth_0_im); axis tight%,pks_0,locs_0);
%                 a.FontSize = 14;
                %findpeaks(line_at_depth_1,x_full * 1e3,'MinPeakHeight',peak_height)
                %subplot(2,4,4);
                %findpeaks(line_at_depth_gs,x_full * 1e3,'MinPeakHeight',peak_height)

%                 subplot(2,3,7);
%                 imagesc(x_full * 1e3,z_field * 1e3,im_1);colormap jet; axis square; axis on; zoom(1);
                clc;
                j
                style = hgexport('factorystyle');
                style.Bounds = 'loose';
                if strcmp(mode,'floats')
                    %export_fig(gcf,['results/figures comparision with db/' int2str(j) ' ' int2str(num_peaks) '.tif'])
                    hgexport(gcf, ['results/figures comparision no db/' int2str(j) ' ' int2str(num_peaks) '.tif'], style, 'Format', 'tiff');
                else
                    %export_fig(gcf,['results/figures comparision integers with db/' int2str(j) ' ' int2str(num_peaks) '.tif'])
                    hgexport(gcf, ['results/figures comparision integers no db/' int2str(j) ' ' int2str(num_peaks) '.tif'], style, 'Format', 'tiff');
                end
                close all
            end
            num_for_each(num_peaks) = num_for_each(num_peaks) + 1;
            if length(pks_gs) == num_peaks
                before = success_gs(num_peaks);
                success_gs(num_peaks) =  before + 1;
            end
            if length(pks_0) == num_peaks
                before = success_0(num_peaks);
                success_0(num_peaks) = before + 1;
            end
%             if length(pks_1) == num_peaks
%                 before = success_1(num_peaks);
%                 success_1(num_peaks) = before + 1;
%             end
            mse_gs(num_peaks) = mse_gs(num_peaks) + sum((line_at_depth_gs - pattern_for_comparision).^2);
           % mse_1(num_peaks) = mse_1(num_peaks) + sum((line_at_depth_1 - pattern_for_comparision).^2);
            mse_0(num_peaks) = mse_0(num_peaks) + sum((line_at_depth_0 - pattern_for_comparision).^2);
            mse_gs_conv(num_peaks) = mse_gs_conv(num_peaks) + sum((line_at_depth_gs - pattern_for_comparision_conv).^2);
            %mse_1_conv(num_peaks) = mse_1_conv(num_peaks) + sum((line_at_depth_1 - pattern_for_comparision_conv).^2);
            mse_0_conv(num_peaks) = mse_0_conv(num_peaks) + sum((line_at_depth_0 - pattern_for_comparision_conv).^2);
        end

    end
        total_results(1,:,i) = success_gs';
        total_results(2,:,i) = success_0';
        total_results(3,:,i) = success_1'; 
        total_amounts(:,i) = num_for_each;
        total_mse(1,:,i) = mse_gs';
        total_mse(2,:,i) = mse_0';
        total_mse(3,:,i) = mse_1';
        total_mse_conv(1,:,i) = mse_gs_conv';
        total_mse_conv(2,:,i) = mse_0_conv';
        total_mse_conv(3,:,i) = mse_1_conv';
    end
end

full.gs = success_gs_total;
full.single_phase = success_0;
full.double_phase = success_1_total;
full.total_results = total_results;
full.total_amount = total_amounts;
full.total_mse = total_mse;
full.total_mse_conv = total_mse_conv;
if strcmp(mode,'floats')
    save 'py to matlab'\'success rate 3 repeats 10,000 each gs,net single,net multi 2 cycles.mat' full
else
    save 'py to matlab'\'success rate 3 repeats 10,000 each gs,net single,net multi integers 2 cycles.mat' full
end
%%
mode = 'integers';
success_name = 'py to matlab'\'success rate 3 repeats 10,000 each gs,net single,net multi 2 cycles.mat';
MSE_conv_image = 'results/results comparision success MSE conv only gs 0 2 cycles.tif';
success_rate_image = 'results/results comparision success only gs 0 2 cycles.tif';
MSE_image = 'results/results comparision success MSE only gs 0 2 cycles.tif';
if strcmp(mode,'integers')
    success_name = 'py to matlab\success rate 3 repeats 10,000 each gs,net single,net multi integers 2 cycles.mat';
    success_rate_image = 'results/results comparision success only gs 0 integers 2 cycles.tif';
    MSE_image = 'results/results comparision success MSE only gs 0 integers 2 cycles.tif';
    MSE_conv_image = 'results/results comparision success MSE conv only gs 0 integers 2 cycles.tif';
end
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
% total_results = full.full.total_mse;
% pct = zeros(2,10,3);
% pct(1,:,:)= squeeze(total_results(1,:,:)) ./ total_amounts;
% pct(2,:,:) = squeeze(total_results(2,:,:)) ./ total_amounts;
% %pct(3,:,:) = squeeze(total_results(3,:,:)) ./ total_amounts;
% means = mean(pct,3)';
% stds = std(pct,0,3)';
% 
% y = squeeze(means);
% error_bar = squeeze(stds);
% figure
% h = plot(y); ylabel('MSE'); xlabel('Number of focuses')
% legend({'GS' 'Model Single Phase' })
% hold on
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

%%
spacing_factor = 2;
pitch = 0.218e-3;
x_min = -17e-3;
x_max = 17e-3;
z_min = 20e-3;
z_max = 60e-3;
x = linspace(x_min,x_max,200);
x_min = -(256 - 1) * pitch/spacing_factor;
x_max = 256 * pitch/spacing_factor;
x_full = x_min:pitch/spacing_factor:x_max;
z_field = linspace(z_min,z_max,300);
N = 1024;
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
range_x = round(18.5e-3 / pitch_half);
x_for_image = (-range_x * pitch_half):pitch_half:range_x*pitch_half;

dz = 0.1e-3;
z_max_gs = 2*40e-3;
z = 0:dz:z_max_gs;
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
num_peaks = 4;
loaded_for_comparision = load(['C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision ' num2str(num_peaks) ' focus.mat']);

peak_height = 0.5;
total_results = zeros(2,19,3);
total_amounts = zeros(19,3);
total_mse = zeros(2,19,3);
total_mse_conv = zeros(2,19,3);

for separation=5:24
    source_sep = loaded_for_comparision.(['sources_' int2str(separation - 5)]);
    base_sep = loaded_for_comparision.(['base_patterns_' int2str(separation - 5)]);
    delays_sep = loaded_for_comparision.(['results_' int2str(separation - 5)]);
    size_to_compute = size(delays_sep,1);
    perm = randperm(size_to_compute);
    step_size = floor(size_to_compute / 3);
    success_gs_sep = zeros(1,3);
    success_net_sep = zeros(1,3);
    mse_gs_sep = zeros(1,3); 
    mse_gs_conv_sep = zeros(1,3);
    mse_net_sep = zeros(1,3);
    mse_net_conv_sep = zeros(1,3);
    num_for_each = ones(1,3) * step_size;
    for i=1:3
        for k = (i-1) * step_size + 1:i * step_size
            j = perm(k);
            base_pattern = base_sep(j,:);
            patterns = padarray(base_pattern,[0 256],0,'both');
            [amps,delays_gs] = calculateGS(patterns,false);
            delays_gs = normalize_delays(delays_gs);
            delays_0 = normalize_delays(delays_sep(j,:));
            line_at_depth_gs = create_new_line(delays_gs,ones(1,128),1,spacing_factor);
            line_at_depth_0 = create_new_line(delays_0,ones(1,128),1,spacing_factor);
            [pks_base,locs_base,widths_base,~] = findpeaks(base_pattern,x_full_gs,'MinPeakHeight',peak_height);
            locs_base = locs_base + abs(x_min);
            locs_base = round(locs_base / (pitch/spacing_factor)) + 1 ;
            pattern_for_comparision = zeros(1,512);
            pattern_for_comparision(locs_base) = pks_base;
            pattern_for_comparision_conv = interp1(x_full_gs,source_sep(j,:),x_full);
            [pks_gs,locs_gs,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
            [pks_0,locs_0,widths_base_0,~] = findpeaks(line_at_depth_0,x_full,'MinPeakHeight',peak_height);
            if length(pks_gs) == 0 || length(pks_0) == 0
                j
            end
        if (num_peaks ~= length(pks_gs) || num_peaks ~= length(pks_0))  && i * k < 10
                im_gs = squeeze(create_new_image(delays_gs));
                im_0 = squeeze (create_new_image(delays_0));
                %im_1 = squeeze(create_new_image(delays_1));

                [amps, new_transducer] = calculateGS(patterns,false);
                new_transducer = amps .* exp(1i * new_transducer);

                new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
                new_transducer = new_transducer(1,256:768 -1);
                new_transducer = imresize(new_transducer,2,'box');

                new_transducer = new_transducer(1,1:N);

                clear E_FSP1_z0
                index = 1;
                for z0 = z
                %     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
                    %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                
                    [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                    index = index + 1;
                    z0/z_max;
                end
                I_FSP1_z0 = abs(E_FSP1_z0).^2;
                line_gs = abs(FSP_X_near(new_transducer,40e-3,N,pitch_half,Wavelength)).^2;
                %line_gs = line_gs(N/2 - 200:N/2 + 200);
                line_gs = interp1(x_half,line_gs,x_full);
                line_gs = line_gs ./ max(max(line_gs));
                I_FSP1_z0 = I_FSP1_z0(:,512 - range_x:512 + range_x);
                [pks_gs_base,locs_gs_base] = extract_peaks(line_gs,pattern_for_comparision);
                figure
                ylabel('Intensity a.u')
                subplot(2,3,1);
                imagesc(x_for_image * 1e3,z * 1e3,I_FSP1_z0); colormap hot; axis square; axis on; zoom(1);
                subplot(2,3,2);
                imagesc(x_full * 1e3,z_field * 1e3,im_gs);colormap hot; axis square; axis on; zoom(1);
                subplot(2,3,3);
                imagesc(x_full * 1e3,z_field * 1e3,im_0);colormap hot; axis square; axis on; zoom(1);
                subplot(2,3,4)
                plot(x_full * 1e3,line_gs)%,pks_gs_base,locs_gs_base);
                %plot(x_full,pattern_for_comparision_conv);
                subplot(2,3,5);
                plot(x_full * 1e3,line_at_depth_gs)%,pks_gs,locs_gs);
                subplot(2,3,6);
                plot(x_full * 1e3,line_at_depth_0)%,pks_0,locs_0);
                %findpeaks(line_at_depth_1,x_full * 1e3,'MinPeakHeight',peak_height)
                %subplot(2,4,4);
                %findpeaks(line_at_depth_gs,x_full * 1e3,'MinPeakHeight',peak_height)

%                 subplot(2,3,7);
%                 imagesc(x_full * 1e3,z_field * 1e3,im_1);colormap jet; axis square; axis on; zoom(1);

                j;
                path_for_figure = ['results/figures comparision patterns no db/' int2str(separation) ' ' int2str(i*k) ' ' int2str(num_peaks) '.tif'];
                hgexport(gcf, [path_for_figure], hgexport('factorystyle'), 'Format', 'tiff');
                close all
            end
            
            if length(pks_gs) == num_peaks
                before = success_gs_sep(i);
                success_gs_sep(i) =  before + 1;
            elseif true
                length(pks_gs); 
            end
        if length(pks_0) == num_peaks
            before = success_net_sep(i);
            success_net_sep(i) = before + 1;
        end
        
         mse_gs_sep(i) = mse_gs_sep(i) + sum((line_at_depth_gs - pattern_for_comparision).^2);
         mse_net_sep(i) = mse_net_sep(i) + sum((line_at_depth_0 - pattern_for_comparision).^2);
         mse_gs_conv_sep(i) = mse_gs_conv_sep(i) + sum((line_at_depth_gs - pattern_for_comparision_conv).^2);
         mse_net_conv_sep(i) = mse_net_conv_sep(i) + sum((line_at_depth_0 - pattern_for_comparision_conv).^2);
        end
    end

    
        total_results(1,separation - 4,:) = success_gs_sep';
        total_results(2,separation - 4,:) = success_net_sep';
        total_amounts(separation - 4,:) = num_for_each';
        total_mse(1,separation - 4,:) = mse_gs_sep';
        total_mse(2,separation - 4,:) = mse_net_sep';
        total_mse_conv(1,separation - 4,:) = mse_gs_conv_sep';
        total_mse_conv(2,separation - 4,:) = mse_net_conv_sep';
end

full.total_results = total_results;
full.total_amount = total_amounts;
full.total_mse = total_mse;
full.total_mse_conv = total_mse_conv;

name_to_save = ['py to matlab/success rate 3 repeats ' num2str(num_peaks) ' focus.mat']; 
save(name_to_save,'full')

num_foci = num_peaks;
success_name = name_to_save;
MSE_conv_image = ['results/results comparision ' num2str(num_foci) ' focus MSE conv.tif'];
success_rate_image = ['results/results comparision ' num2str(num_foci) ' focus success rate.tif'];
MSE_image = ['results/results comparision ' num2str(num_foci) ' focus MSE.tif'];
x_separation = 5 * pitch*1e3:pitch * 1e3:24 * pitch * 1e3;
full = load(success_name);
total_results = full.full.total_results;
total_amounts = full.full.total_amount;
pct = zeros(2,20,3);
pct(1,:,:)= squeeze(total_results(1,:,:)) ./ total_amounts;
pct(2,:,:) = squeeze(total_results(2,:,:)) ./ total_amounts;
%pct(3,:,:) = squeeze(total_results(3,:,:)) ./ total_amounts;
means = mean(pct,3)';
stds = std(pct,0,3)';

y = squeeze(means);
error_bar = squeeze(stds);
figure
h = plot(x_separation,y); ylabel('Success rate %','FontSize',16); xlabel('Separation between foci [mm]','FontSize',16)
legend({'GS' 'MultiLevelNet'})
hold on
errorbar([x_separation; x_separation]',y,error_bar,'k','linestyle','none','HandleVisibility','off');
% 
grid on
hgexport(gcf, [success_rate_image], hgexport('factorystyle'), 'Format', 'tiff');

total_results = full.full.total_mse_conv;
pct = zeros(2,20,3);
pct(1,:,:)= squeeze(total_results(1,:,:)) ./ total_amounts;
pct(2,:,:) = squeeze(total_results(2,:,:)) ./ total_amounts;
%pct(3,:,:) = squeeze(total_results(3,:,:)) ./ total_amounts;
means = mean(pct,3)';
stds = std(pct,0,3)';

y = squeeze(means);
error_bar = squeeze(stds);
figure
h = plot(x_separation,y); ylabel('MSE','FontSize',16); xlabel('Separation between foci [mm]','FontSize',16)
legend({'GS' 'MultiLevelNet' })
hold on
errorbar([x_separation; x_separation]',y,error_bar,'k','linestyle','none','HandleVisibility','off');
% 
grid on
hgexport(gcf, [MSE_conv_image], hgexport('factorystyle'), 'Format', 'tiff');

%%
success_gs_total = [];
success_gs_multi_total = [];
success_1_total = [];
pitch = 0.218e-3;
x_min = -(256 - 1) * pitch/5;
x_max = 256 * pitch/5;
x_full = x_min:pitch/5:x_max;

loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision1.mat');
delays_base = loaded_for_comparision.from_gs;
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
success_gs = 0;
over_count_gs = 0;
success_gs_multi = 0;
over_count_gs_multi = 0;
over_count_1 = 0;
success_1 = 0;
num_successful = 0;
peak_height = 0.5;
parfor i=1:length(delays_base)
    init_field
    patterns = padarray(source(i,:),[0 256],0,'both');
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    delays_1 = normalize_delays(delays_step_1(i,:));
    line_at_depth_gs = create_new_line(delays_gs);
    line_at_depth_gs_multi = create_new_line(delays_gs,amps,10);
    line_at_depth_1 = create_new_line(delays_1);
    [pks_gs,~,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
    [pks_gs_multi,~,widths_gs_multi,~] = findpeaks(line_at_depth_gs_multi,x_full,'MinPeakHeight',peak_height);
    [pks_1,~,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);
    [pks_base,~,widths_base,~] = findpeaks(source(i,:),x_full,'MinPeakHeight',peak_height);
    if length(pks_gs) == length(pks_base)
        success_gs = success_gs + 1;
    end
    if length(pks_gs_multi) == length(pks_base)
        success_gs_multi = success_gs_multi + 1;
    end
    if length(pks_1) == length(pks_base)
        success_1 = success_1 + 1;
    end

end
success_gs_total = [success_gs_total success_gs];
success_1_total = [success_1_total success_1];
success_gs_multi_total = [success_gs_multi_total success_gs_multi];
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision2.mat');
delays_base = loaded_for_comparision.from_gs;
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
success_gs = 0;
over_count_gs = 0;
success_gs_multi = 0;
over_count_gs_multi = 0;
over_count_1 = 0;
success_1 = 0;
num_successful = 0;
peak_height = 0.5;
parfor i=1:length(delays_base)
    init_field
    patterns = padarray(source(i,:),[0 256],0,'both');
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    delays_1 = normalize_delays(delays_step_1(i,:));
    line_at_depth_gs = create_new_line(delays_gs);
    line_at_depth_gs_multi = create_new_line(delays_gs,amps,10);
    line_at_depth_1 = create_new_line(delays_1);
    [pks_gs,~,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
    [pks_gs_multi,~,widths_gs_multi,~] = findpeaks(line_at_depth_gs_multi,x_full,'MinPeakHeight',peak_height);
    [pks_1,~,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);
    [pks_base,~,widths_base,~] = findpeaks(source(i,:),x_full,'MinPeakHeight',peak_height);
    if length(pks_gs) == length(pks_base)
        success_gs = success_gs + 1;
    end
    if length(pks_gs_multi) == length(pks_base)
        success_gs_multi = success_gs_multi + 1;
    end
    if length(pks_1) == length(pks_base)
        success_1 = success_1 + 1;
    end

end
success_gs_total = [success_gs_total success_gs];
success_1_total = [success_1_total success_1];
success_gs_multi_total = [success_gs_multi_total success_gs_multi];
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision3.mat');
delays_base = loaded_for_comparision.from_gs;
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
success_gs = 0;
over_count_gs = 0;
success_gs_multi = 0;
over_count_gs_multi = 0;
over_count_1 = 0;
success_1 = 0;
num_successful = 0;
peak_height = 0.5;
parfor i=1:length(delays_base)
    init_field
    patterns = padarray(source(i,:),[0 256],0,'both');
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    delays_1 = normalize_delays(delays_step_1(i,:));
    line_at_depth_gs = create_new_line(delays_gs);
    line_at_depth_gs_multi = create_new_line(delays_gs,amps,10);
    line_at_depth_1 = create_new_line(delays_1);
    [pks_gs,~,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
    [pks_gs_multi,~,widths_gs_multi,~] = findpeaks(line_at_depth_gs_multi,x_full,'MinPeakHeight',peak_height);
    [pks_1,~,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);
    [pks_base,~,widths_base,~] = findpeaks(source(i,:),x_full,'MinPeakHeight',peak_height);
    if length(pks_gs) == length(pks_base)
        success_gs = success_gs + 1;
    end
    if length(pks_gs_multi) == length(pks_base)
        success_gs_multi = success_gs_multi + 1;
    end
    if length(pks_1) == length(pks_base)
        success_1 = success_1 + 1;
    end

end
success_gs_total = [success_gs_total success_gs];
success_1_total = [success_1_total success_1];
success_gs_multi_total = [success_gs_multi_total success_gs_multi];
%%
full.gs = success_gs_total;
full.gs_multi = success_gs_multi_total;
full.floats = success_1_total;
save 'py to matlab'\'success rate 3 repeats 5000 each floats.mat' full
%%
success_gs_total = [];
success_gs_multi_total = [];
success_1_total = [];
success_0_total = [];
pitch = 0.218e-3;
x_min = -(256 - 1) * pitch/5;
x_max = 256 * pitch/5;
x_full = x_min:pitch/5:x_max;
x = -(256-1)*pitch:pitch:256*pitch;
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision floats.mat');
delays_base = loaded_for_comparision.from_gs;
inputs = loaded_for_comparision.x_acc;
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
peaks_gs = [];
peaks_gs_multi = [];
peaks_0 = [];
peaks_1 = [];
difference_gs = [];
difference_gs_multi = [];
mse_gs = [];
mse_gs_multi = [];
difference_0 = [];
mse_0 = [];
difference_1 = [];
mse_1 = [];
mse_gs = [];
mse_0 = [];
mse_1 = [];
success_gs = 0;
over_gs = 0;
success_gs_multi = 0;
over_gs_multi = 0;
success_0 = 0;
over_0 = 0;

success_1 = 0;
over_1 = 0;

num_successful = 0;
peak_height = 0.5;
parfor i=1:length(delays_base)
    if i< 1000
        init_field
    end
    patterns = padarray(source(i,:),[0 256],0,'both');
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    delays_0 = normalize_delays(delays_step_0(i,:));
    delays_1 = normalize_delays(delays_step_1(i,:));
    line_at_depth_gs = create_new_line(delays_gs);
    line_at_depth_gs_multi = create_new_line(delays_gs,amps,10);
    %line_at_depth_0 = create_new_line(delays_0);
    line_at_depth_1 = create_new_line(delays_1);
    line_interpolated = interp1(x,inputs(i,:),x_full);
    mse_gs = [mse_gs (line_at_depth_gs - line_interpolated.^2)];
    mse_gs_multi = [mse_gs_multi (line_at_depth_gs_multi - line_interpolated.^2)];
    %mse_0 = [mse_0 (line_at_depth_0 - line_interpolated.^2)];
    mse_1 = [mse_1 (line_at_depth_1 - line_interpolated.^2)];
    [pks_gs,~,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
    [pks_gs_multi,~,widths_base_gs_multi,~] = findpeaks(line_at_depth_gs_multi,x_full,'MinPeakHeight',peak_height);
    %[pks_0,~,widths_base_0,~] = findpeaks(line_at_depth_0,x_full,'MinPeakHeight',peak_height);
    [pks_1,~,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);
    [pks_base,~,widths_base,~] = findpeaks(source(i,:),x_full,'MinPeakHeight',peak_height);
    if length(pks_gs) == length(pks_1) && length(pks_gs) == length(pks_base) && length(pks_gs_multi) == length(pks_base)
        peaks_gs = [peaks_gs pks_gs];
        %peaks_0 = [peaks_0 pks_0];
        peaks_1 = [peaks_1 pks_1];
        peaks_gs_multi = [peaks_gs_multi pks_gs_multi];
        num_successful = num_successful + 1;
        difference_gs = [difference_gs (pks_gs - pks_base)];
        difference_gs_multi = [difference_gs_multi (pks_gs - pks_base)];
        %difference_0 = [difference_0 (pks_0 - pks_base)];
        difference_1 = [difference_1 (pks_1 - pks_base)];
    
    end
    if length(pks_gs) == length(pks_base)
        success_gs = success_gs + 1;
    elseif length(pks_gs) > length(pks_base)
        over_gs = over_gs + 1;
    end
    if length(pks_gs_multi) == length(pks_base)
        success_gs_multi = success_gs_multi + 1;
    elseif length(pks_gs_multi) > length(pks_base)
        over_gs_multi = over_gs_multi + 1;
    end
%     if length(pks_0) == length(pks_base)
%         success_0 = success_0 + 1;
%     elseif length(pks_0) > length(pks_base)
%         over_0 = over_0 + 1;
%     end
    if length(pks_1) == length(pks_base)
        success_1 = success_1 + 1;
    elseif length(pks_1) > length(pks_base)
        over_1 = over_1 + 1;
    end

end

to_save.peaks_gs = peaks_gs;
to_save.peaks_1 = peaks_1;
to_save.peaks_gs_multi = peaks_gs_multi;
to_save.difference_gs = difference_gs;
to_save.difference_1 = difference_1;
to_save.difference_gs_multi = difference_gs_multi;
to_save.success_gs = success_gs;
to_save.success_1 = success_1;
to_save.success_gs_multi = success_gs_multi;
to_save.mse_gs = mse_gs;
to_save.mse_gs_multi = mse_gs_multi;
to_save.mse_1 = mse_1;
save 'py to matlab'\'comparision 15000 examples floats.mat'
%%
x = categorical({'Average amplitude of focus intensity' 'Average deviation of focus intensity'});
y = [mean(peaks_gs) mean(peaks_1) mean(peaks_gs_multi) ;
    mean(abs(difference_gs))  mean(abs(difference_1)) mean(abs(difference_gs_multi))];
error_bar = [std(peaks_gs) std(peaks_1) std(peaks_gs_multi) ;
    std(difference_gs) std(difference_1) std(difference_gs_multi)];
num_successful
10 * log10(mean(peaks_0)/mean(peaks_gs))
figure
h = bar(x,y); title('Average amplitude and deviation of focuses','FontSize',16); ylabel('Intensity [a.u]');
%set(h, {'DisplayName'}, {'Output from Gershberg-Saxton' 'Output from model' 'Output from Gershberg-Saxton with multi-cycle'})
legend({'Output from Gerchberg-Saxton single-cycle' 'Output from model' 'Output from Gerchberg-Saxton multi-cycle'},'FontSize',14,'FontName','roman')

hold on
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(y)
% Get the x coordinate of the bars
x_new = nan(nbars, ngroups);
for i = 1:nbars
    h(i).XEndPoints

    x_new(i,:) = h(i).XEndPoints';
end
oldy = get(gca, 'ylim')
if oldy(1) > '0'
    oldy = [0 oldy(2) * 2]
end
set(gca,'YLim',[0 1.2])
% Plot the errorbars
errorbar(x_new',y,error_bar,'k','linestyle','none','HandleVisibility','off');

%er = errorbar(x,y,error_bar);
% figure
% y_difference = [mean(difference_gs.^2) mean(difference_0.^2) mean(difference_1.^2) ];
% bar(x,y_difference); title('average deviation from target focus amplitude')
%%
load 'py to matlab'\'success rate 3 repeats 5000 each floats.mat'
success_gs_total = full.gs;
success_1_total = full.floats;
success_gs_multi_total = full.gs_multi;
figure
grid on
x = categorical({'From Gerchberg-Saxton single cycle' 'From model' 'From Gerchberg-Saxton multi-cycle'});
x = reordercats(x,cellstr(x)');

y_success = [mean(success_gs_total/5000) mean(success_1_total/5000) mean(success_gs_multi_total/5000) ];
std_success = [std(success_gs_total/5000) std(success_gs_multi_total/5000) std(success_1_total/5000)];

h = bar(x,diag(y_success),'stacked'); title('percent of successfull attemps','FontSize',16,'FontName','Times New Roman'); ylabel('% of success','FontSize',12,'FontName','Times New Roman');
set(gca,'FontSize',12,'FontName','Times New Roman')
hold on
errorbar(x,y_success,std_success,'k','LineStyle','none')
%%
init_field
separation_factor = 1.5;
x_min = -(256 - 1) * pitch/separation_factor;
x_max = 256 * pitch/separation_factor;
x_full = x_min:pitch/separation_factor:x_max;
x_min = -17e-3;
x_max = 17e-3;
z_min = 20e-3;
z_max = 60e-3;
x = linspace(x_min,x_max,200);
dz = 0.1e-3;
z_max_gs = 2*40e-3;
z = 20e-3:dz:60e-3;

z_field = linspace(z_min,z_max,300);
N = 1024;
pitch = 0.218e-3;

pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
range_x = round(17e-3 / pitch_half);
x_for_image = (-range_x * pitch_half):pitch_half:range_x*pitch_half;
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
x_pitch = -256 * pitch:pitch:(256 - 1) * pitch;
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\comparision effective diffraction.mat');
source = double(loaded_for_comparision.source);
base = double(loaded_for_comparision.base_patterns);
%delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = double(loaded_for_comparision.from_net_step_0);
num_focus = 5;
source = squeeze(source(num_focus - 1,:,:));
base = squeeze(base(num_focus - 1,:,:));
delays_step_0 = squeeze(delays_step_0(num_focus - 1,:,:));
separations = [10 15 20];
% fig = figure;
% fig;
% fig.Position = [100 100 1280 1020];
for i=1:3
    sep = separations(i);
    base_pattern = base(sep - 1,:);
    base_pattern = reshape(base_pattern,[1,512]);
    patterns = padarray(base_pattern,[0 256],0,'both');
    delays_model = delays_step_0(sep - 1,:);
    delays_model = reshape(delays_model,[1,128]);
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    delays_model = normalize_delays(delays_model);
    im_gs = squeeze(create_new_image(delays_gs,amps,1));
    im_0 = squeeze(create_new_image(delays_model,amps,1));
    [amps, new_transducer] = calculateGS(patterns,true);
    new_transducer = amps .* exp(1i * new_transducer);

    new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
    new_transducer = new_transducer(1,256:768 -1);
    new_transducer = imresize(new_transducer,2,'box');

    new_transducer = new_transducer(1,1:N);

    clear E_FSP1_z0
    index = 1;
    for z0 = z
    %     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
        %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
    
        [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
        index = index + 1;
        z0/z_max;
    end
    I_FSP1_z0 = abs(E_FSP1_z0).^2;
    I_FSP1_z0 = I_FSP1_z0 ./ max(max(I_FSP1_z0));
    I_FSP1_z0 = I_FSP1_z0(:,512 - range_x:512 + range_x);
    I_FSP1_z0 = squeeze(create_new_image(delays_gs,amps,100));
    fig = figure;
    %fig.Position = [100 100 1280 * 1 1020 * 1];
    %a = subplot(3,4,1 + 4 * (i-1));
    %imagesc(x_for_image * 1e3,z * 1e3,I_FSP1_z0); colormap hot; axis square; axis on; zoom(1);
    a = subplot(2,2,1);
    imagesc(x_full * 1e3,z * 1e3,I_FSP1_z0); colormap hot; axis square; axis on; zoom(1);
    %a = gca;
    if true
        xlabel('x [mm]','FontSize',26); ylabel('z [mm]','FontSize',26);
        a.FontSize = 26;
        pos1 = get(a,'Position');
        colorbar('eastoutside')
        set(a,'Position',pos1)
    else
        set(gca,'xticklabel',[])
        set(gca,'yticklabel',[])
    end
    %hgexport(gcf, ['results/images for pattern/CW ' int2str(sep) '.tif'], style, 'Format', 'tiff');
    %subplot(3,4,2 + 4 * (i-1))
    a = subplot(2,2,2);
    %figure
    imagesc(x * 1e3,z_field * 1e3,im_gs);colormap hot; axis square; axis on; zoom(1);
    %set(gca,'xticklabel',[])
    %set(gca,'yticklabel',[])
    %a = gca;
    xlabel('x [mm]','FontSize',26); ylabel('z [mm]','FontSize',26);
    a.FontSize = 26;
    pos1 = get(a,'Position');
    colorbar('eastoutside')
    set(a,'Position',pos1)
    %hgexport(gcf, ['results/images for pattern/GS ' int2str(sep) '.tif'], style, 'Format', 'tiff');


    %figure
    a = subplot(2,2,3);
    imagesc(x * 1e3,z_field * 1e3,im_0);colormap hot; axis square; axis on; zoom(1);
%     set(gca,'xticklabel',[])
%     set(gca,'yticklabel',[])
%     a = gca;
    xlabel('x [mm]','FontSize',26); ylabel('z [mm]','FontSize',26);
    a.FontSize = 26;
    pos1 = get(a,'Position');
    colorbar('eastoutside')
    set(a,'Position',pos1)
    
    %hgexport(gcf, ['results/images for pattern/USDL ' int2str(sep) '.tif'], style, 'Format', 'tiff');

    %figure
    a = subplot(2,2,4);
    [amps, delays_gs] = calculateGS(patterns,false);
    delays_gs = normalize_delays(delays_gs);
    line = im_gs(round(size(im_gs,1)/2),:);
    line = line - min(line);
    line = line ./ max(line);
    line = create_new_line(delays_gs,amps,1,separation_factor);
    %[~,~,widths,~] = findpeaks(line,x_full,'MinPeakHeight',0.5,'WidthReference','halfheight');    %line = line(256 - range_x:256 + range_x);
    %widths
    %means_gs = mean(widths)
    plot(x_full * 1e3,line,'LineWidth',2); axis tight square; axis on;
    %a = gca;    
    a.FontSize = 26;
    xlabel('x [mm]','FontSize',26); ylabel('Intensity [a.u]', FontSize=26);
    hold on
    line = im_0(round(size(im_0,1)/2),:);
    line = line - min(line);
    line = line ./ max(line);
    line = create_new_line(delays_model,amps,1,separation_factor);
    %[~,~,widths,~] = findpeaks(line,x_full,'MinPeakHeight',0.5,'MinPeakProminence',0.2,'WidthReference','halfheight');    %line = line(256 - range_x:256 + range_x);
    %widths
    %means_net = mean(widths)
    %line = line(256 - range_x:256 + range_x);
    plot(x_full * 1e3,line,'LineWidth',2);
    %hgexport(gcf, ['results/images for pattern/cross section ' int2str(sep) '.tif'], style, 'Format', 'tiff');
    legend({'GS 1-cycle','USDL'})
    close all
end
% fig = figure;
% for i=1:8
%     sep = separations(i);
%     base_pattern = source(sep - 1,:);
%     base_pattern = reshape(base_pattern,[1,512]);
%     patterns = padarray(base_pattern,[0 256],0,'both');
%     delays_model = delays_step_0(sep - 1,:);
%     delays_model = reshape(delays_model,[1,128]);
%     [amps, delays_gs] = calculateGS(patterns,false);
%     delays_gs = normalize_delays(delays_gs);
%     delays_model = normalize_delays(delays_model);
%     line_0 = create_new_line(delays_model,amps,1,separation_factor);
%     line_gs_at_depth = create_new_line(delays_gs,amps,1,separation_factor);
%     [amps, new_transducer] = calculateGS(patterns,false);
%     new_transducer = amps .* exp(1i * new_transducer);
% 
%     new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
%     new_transducer = new_transducer(1,256:768 -1);
%     new_transducer = imresize(new_transducer,2,'box');
% 
%     new_transducer = new_transducer(1,1:N);
% 
%     clear E_FSP1_z0
%     line_gs = abs(FSP_X_near(new_transducer,40e-3,N,pitch_half,Wavelength)).^2;
% 
%     %line_gs = line_gs(N/2 - 200:N/2 + 200);
%     line_gs = interp1(x_half,line_gs,x_full);
%     line_gs = line_gs ./ max(max(line_gs));
%     subplot(4,8,i)
%     plot(x_full * 1e3,line_gs); axis tight
%     subplot(4,8,8 + i)
%     plot(x_full * 1e3,line_gs_at_depth); axis tight
%     subplot(4,8,16 + i)
%     plot(x_full * 1e3,line_0); axis tight
%     subplot(4,8,24 + i)
%     plot(x_pitch(256 - 70:256 + 70),base(sep - 1, 256 - 70:256 + 70))
% end
%%
init_field
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\comparision effective diffraction.mat');
source = double(loaded_for_comparision.source);
base = double(loaded_for_comparision.base_patterns);
%delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = double(loaded_for_comparision.from_net_step_0);
%delays_step_1 = loaded_for_comparision.from_net_step_1;
effective_thresh = zeros(2,9);

pitch = 0.218e-3;
x_min = -(256 - 1) * pitch;
x_max = 256 * pitch;

x_full = x_min:pitch:x_max;
x_min = -(256 - 1) * pitch/3;
x_max = 256 * pitch/3;

x_line = x_min:pitch/3:x_max;
peak_thresh = 0.5;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
results_success = zeros(2,9,25);
for num_focus=2:10
    for separation=2:29
        base_pattern = source(num_focus - 1,separation - 1,:);
        base_pattern = reshape(base_pattern,[1,512]);
        patterns = padarray(base_pattern,[0 256],0,'both');
        delays_model = delays_step_0(num_focus - 1, separation - 1,:);
        delays_model = reshape(delays_model,[1,128]);
        [amps, delays_gs] = calculateGS(patterns,false);
        delays_gs = normalize_delays(delays_gs);
        delays_model = normalize_delays(delays_model);
        fut = parfeval(@create_new_line, 1, delays_gs);
        
        % Block for up to maxTime seconds waiting for a result
        didFinish = wait(fut, 'finished', 30);
        if ~didFinish
        % Execution didn't finish in time, cancel this iteration
            cancel(fut);
            line_gs = zeros(1,512);
        else
        % Did complete, retrieve results
            line_gs = fetchOutputs(fut);
        end
        fut = parfeval(@create_new_line, 1, delays_model);
        %line_model = create_new_line(delays_model);
        % Block for up to maxTime seconds waiting for a result

        didFinish = wait(fut, 'finished', 30);
        if ~didFinish
        % Execution didn't finish in time, cancel this iteration
            cancel(fut);
            line_model = zeros(1,512);
        else
        % Did complete, retrieve results
            line_model = fetchOutputs(fut);
        end
        %line_gs = create_new_line(delays_gs);
        %line_model = create_new_line(delays_model);
        [pks_base,locs_base,widths_base,~] = findpeaks(base_pattern,x_full_gs,'MinPeakHeight',peak_thresh);
        [pks_gs,~,~,~] = findpeaks(line_gs,x_line,'MinPeakHeight',peak_thresh);%,'MinPeakDistance',separation * pitch / 2,'MinPeakProminence',0.2);
        [pks_model,~,~,~] = findpeaks(line_model,x_line,'MinPeakHeight',peak_thresh);%,'MinPeakDistance',separation * pitch / 2,'MinPeakProminence',0.2);
        if length(pks_gs) == num_focus
            results_success(1,num_focus - 1,separation - 1) = 1;
        end
        if length(pks_model) == num_focus
            results_success(2,num_focus - 1,separation - 1) = 1;
        end  
    end
end

results_gs = squeeze(results_success(1,:,:));
results_model = squeeze(results_success(2,:,:));
[~,minimums_gs] = max(results_gs,[],2);
[~,minimums_net] = max(results_model,[],2);
[~,maximums_gs] = max(flip(results_gs,2),[],2);
[~,maximums_net] = max(flip(results_model,2),[],2);
maximums_gs = 29 - maximums_gs;
maximums_net = 29 - maximums_net;
x_range = 2:10;
separation_range = pitch * 2:19;
figure
plot(x_range,minimums_gs * pitch * 1e3); ylabel('separation between foci [mm]'); xlabel('number of foci')
hold on
plot(x_range,minimums_net * pitch * 1e3)
legend({'GS','model'})
hgexport(gcf, ['results/minimal distance plot.tif'], hgexport('factorystyle'), 'Format', 'tiff');
figure
plot(x_range,maximums_gs * pitch * 1e3); ylabel('separation between foci [mm]'); xlabel('number of foci')
hold on
plot(x_range,maximums_net * pitch * 1e3)
legend({'GS','model'})
hgexport(gcf, ['results/maximal distance plot.tif'], hgexport('factorystyle'), 'Format', 'tiff');
%%
num_focus = 4;
separation = 3;
src = base(num_focus - 1,separation - 1,:);
base_pattern = source(num_focus - 1,separation - 1,:);
delays_model = delays_step_0(num_focus - 1, separation - 1,:);
delays_model = reshape(delays_model,[1,128]);
delays_model = normalize_delays(delays_model);

base_pattern = reshape(base_pattern,[1,512]);
patterns = padarray(base_pattern,[0 256],0,'both');
[amps, delays_gs] = calculateGS(patterns,false);
delays_gs = normalize_delays(delays_gs);
line = create_new_line(delays_gs);
figure
plot(x_full,squeeze(src))
hold on
findpeaks(line,x_line,'MinPeakHeight',peak_thresh,'MinPeakDistance',separation * pitch / 2,'MinPeakProminence',0.2)
%%

function plot_peaks(x,y,peaks,locs)
    plot(x,y);
    hold on
    scatter(x(locs),peaks + 0.05,[],'v','MarkerFaceColor',[0,0,0],'MarkerEdgeColor',[0,0,0])
end
function [peaks,locs] = extract_peaks(input, pattern)
    pitch = 0.218e-3;
    x_min = -(256 - 1) * pitch/3;
    x_max = 256 * pitch/3;
    x_full = x_min:pitch/3:x_max;
    peaks = [];
    locs = [];
    peak_height = 0.5;
    [pks_base,locs_base,~,~] = findpeaks(pattern,'MinPeakHeight',peak_height);
    [pks_input,locs_input,~,~] = findpeaks(input,'MinPeakHeight',peak_height/2,'MinPeakDistance',2);
    j = 1;
    for i=1:length(pks_input)
        if j > length(pks_base) && pks_input(i) > peak_height
            peaks = [];
            locs = [];
            break
        elseif j > length(pks_base)
            continue
        end
        if  pks_base(j)/2 < pks_input(i) && abs(locs_base(j) - locs_input(i)) < 30
            peaks = [peaks pks_input(i)];
            locs = [locs locs_input(i)];
            j = j + 1;
        elseif pks_input(i) > peak_height
            peaks = [];
            locs = [];
            break
        end
    end
end