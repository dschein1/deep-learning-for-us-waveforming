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
clu = parcluster;
opts = parforOptions(clu);
parfor (i=1:16,opts)
    init_field
end
success_gs_total = [];
success_gs_multi_total = [];
success_1_total = [];
pitch = 0.218e-3;
x_min = -(256 - 1) * pitch/5;
x_max = 256 * pitch/5;
x_full = x_min:pitch/5:x_max;
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision with base model.mat');
delays_base = loaded_for_comparision.from_gs;
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
total_size = length(delays_step_1);
size_of_step = floor(total_size / 3);
peak_height = 0.5;
total_results = zeros(3,10,3);
for i=1:3
    success_gs = zeros(1,10);
    over_count_gs = 0;
    success_0 = zeros(1,10);
    over_count_gs_multi = 0;
    over_count_1 = 0;
    success_1 = zeros(1,10);
    num_for_each = zeros(10,1);
    for j=(i-1) * size_of_step + 1:i *size_of_step
        patterns = padarray(source(j,:),[0 256],0,'both');
        [amps, delays_gs] = calculateGS(patterns,false);
        delays_gs = normalize_delays(delays_gs);
        delays_0 = normalize_delays(delays_step_0(j,:));
        delays_1 = normalize_delays(delays_step_1(j,:));
        line_at_depth_gs = create_new_line(delays_gs);
        line_at_depth_0 = create_new_line(delays_0);
        line_at_depth_1 = create_new_line(delays_1);
        [pks_gs,~,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
        [pks_0,~,widths_base_0,~] = findpeaks(line_at_depth_0,x_full,'MinPeakHeight',peak_height);
        [pks_1,~,widths_base_1,~] = findpeaks(line_at_depth_1,x_full,'MinPeakHeight',peak_height);
        [pks_base,~,widths_base,~] = findpeaks(source(i,:),x_full,'MinPeakHeight',peak_height);
        num_peaks = length(pks_base);
        num_for_each(num_peaks) = num_for_each(num_peaks) + 1;
        if length(pks_gs) == num_peaks
            before = success_gs(num_peaks);
            success_gs(num_peaks) =  before + 1;
        end
        if length(pks_0) == num_peaks
            before = success_0(num_peaks);
            success_0(num_peaks) = before + 1;
        end
        if length(pks_1) == num_peaks
            before = success_1(num_peaks);
            success_1(num_peaks) = before + 1;
        end

    end
        total_results(1,:,i) = success_gs';
        total_results(2,:,i) = success_0';
        total_results(3,:,i) = success_1';
end

full.gs = success_gs_total;
full.gs_multi = success_0;
full.floats = success_1_total;
save 'py to matlab'\'success rate 3 repeats 10,000 each gs,net single,net multi.mat' full

means = mean(total_results,2);
stds = std(total_results,0,2);

y = means;
error_bar = stds;
figure
h = bar(y); ylabel('Success rate %'); xlabel('Number of focuses')
%set(h, {'DisplayName'}, {'Output from Gershberg-Saxton' 'Output from model' 'Output from Gershberg-Saxton with multi-cycle'})
legend('GS' 'Model Single Phase' 'Model Double Phase')
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
% 
grid on
set(gca,'xTick',[-5:2.5:+5])
set(gca,'YTick',[-dB:+dB/2:0])
set(gca,'xTickLabel',[])
set(gca,'YTickLabel',[])
hgexport(gcf, ['Sim_PSF_Lateral',num2str(dB),'dB.tif'], hgexport('factorystyle'), 'Format', 'tiff');
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
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\comparision effective diffraction.mat');
source = loaded_for_comparision.source;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_0 = loaded_for_comparision.from_net_step_0;
delays_step_1 = loaded_for_comparision.from_net_step_1;
effective_thresh = zeros(3,8);

pitch = 0.218e-3;
x_min = -(256 - 1) * pitch/5;


x_max = 256 * pitch/5;
x_full = x_min:pitch/5:x_max;
peak_thresh = 0.5;
for num_focus=2:10
    separation=2;
    gs_found = false;
    gs_multi_found = false;
    model_found = false;
    while separation<12 && ~gs_found && ~gs_multi_found && ~model_found
        %patterns = padarray(source(i,:),[0 256],0,'both');
        temp = source(num_focus - 1,separation - 1,:);
        temp = reshape(temp,1,512);
        base = padarray(temp,[0 256],0,'both');

        [amps, delays_gs] = calculateGS(base,false);
        delays_gs = normalize_delays(delays_gs);
        if ~gs_found
            line = create_new_line(delays_gs);
            [pks,~,~,~] = findpeaks(line,x_full,'MinPeakHeight',peak_thresh);
            if length(pks) == num_focus
                gs_found = true;
                effective_thresh(1,num_focus - 1) = separation;
            end
        end
        if ~gs_multi_found
            line = create_new_line(delays_gs,amps,10);
            [pks,~,~,~] = findpeaks(line,x_full,'MinPeakHeight',peak_thresh);
            if length(pks) == num_focus
                gs_multi_found = true;
                effective_thresh(2,num_focus - 1) = separation;
            end
        end
        if ~model_found
            delays = double(delays_step_1(num_focus - 1,separation - 1,:));
            delays = normalize_delays(delays);
            line = create_new_line(delays);
            [pks,~,~,~] = findpeaks(line,x_full,'MinPeakHeight',peak_thresh);
            if length(pks) == num_focus
                model_found = true;
                effective_thresh(3,num_focus - 1) = separation;
            end
        end
        separation = separation + 1;
    end
end
plot(effective_thresh)
