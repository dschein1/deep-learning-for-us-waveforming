init_field

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
findpeaks(line_1,x_full,'MinPeakProminence',0.05,MinPeakHeight=0.2)
[pks,locs,widths_1,proms] = findpeaks(line_1,x_full,'MinPeakProminence',0.3); title('peaks from net step 1')
widths_1

mean(widths_base - widths_1)

%%
loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision.mat');
delays_base = loaded_for_comparision.from_gs;
delays_net_base = loaded_for_comparision.from_net_base;
delays_step_1 = loaded_for_comparision.from_net_step_1;
for i=1:length(delays_base)
    
end