init_field
%close all
clear all
pitch = 0.218e-3;
delays_calc = calc_delay(128,pitch,1490,[0 0 40]/1000); 
pattern= zeros(1,1024);
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
patterns = zeros(1,1024);
%patterns(490) = 1;
%patterns(500) = 1;
patterns(512 - 6) = 1;
patterns(512 + 6) = 1;
patterns(512) = 1;
patterns(512 + 12) = 1;
patterns(512 - 12) = 1;

[amps, delays_gs] =  calculateGS(patterns,false);
i = 1;
loaded = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\single_data1 5 focus from net.mat');
pattern = loaded.source(i,:);
patterns = padarray(pattern,[0 256],0,'both');
[amps, delays_gs] = calculateGS(patterns,false);
[amps_gs, delays_gs_amps] = calculateGS(patterns,true);
delays_gs_amps = normalize_delays(delays_gs_amps);
delays = double(loaded.from_net_base(i,:));
delays_0 = double(loaded.from_net_step_0(i,:));
delays_0 = normalize_delays(delays_0);
delays_1 = double(loaded.from_net_step_1(i,:));
delays_1 = normalize_delays(delays_1);
%delays = delays_gs;
amps = ones(1,128);
%  pattern = generate_patterns(1,1024,6);
%  pattern = patterns(1,:);
% 
% [amps, delays] = calculateGS(pattern,true);

new_transducer = amps .* exp(1i * delays);
 delays_gs = delays_gs - min(min(delays_gs));
delays_gs = unwrap(delays_gs);
delays_gs = delays_gs / (2*pi);
delays_gs = delays_gs / Frequancy;

 delays = delays - min(min(delays));
delays = unwrap(delays);
delays = delays / (2*pi);
delays = delays / Frequancy;
% figure
% subplot(1,2,1)
% plot(delays_gs); title('delays from gs');
% subplot(1,2,2)
%  plot(delays); title('delays from model');
%  delays = delays - min(min(delays));
% delays = unwrap(delays);
% delays = delays / (2*pi);
% delays = delays / Frequancy;
 
% subplot(1,3,3)
% plot(delays ./ delays_calc); title('delays from GS divided by delays from calculation')


im_from_calculated = squeeze(create_new_image(delays_gs));
im_from_field = squeeze(create_new_image(delays,amps));
im_from_net_0 = squeeze(create_new_image(delays_0));
im_from_net_1 = squeeze(create_new_image(delays_1));
im_from_gs_multi_cycle = squeeze(create_new_image(delays_gs,amps,10));
% figure
% subplot(2,2,1)
% imagesc(x * 1e3,z_field * 1e3,im_from_calculated); colormap jet; axis square; axis on; zoom(1); title('from field ii based on calculation')
% subplot(2,2,2)    
% imagesc(x * 1e3,z_field * 1e3,im_from_field); colormap jet; axis square; axis on; zoom(1); title('from field ii based on gs')
% subplot(2,2,3)
% plot(x * 1e3,im_from_calculated(round(40e-3 / dz_field - 10e-3/dz_field),:)); colormap jet; axis square; axis on; zoom(1); title('from field ii at depth 40 mm based on calculation')
% subplot(2,2,4)
% plot(x * 1e3,im_from_field(round(40e-3 / dz_field - 10e-3/dz_field),:)); colormap jet; axis square; axis on; zoom(1); title('from field ii at depth 40 mm based on gs')

N = 1024;
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full = -(512-1)*pitch/2:pitch:512*pitch/2;
dz = 0.1e-3;
z_max = 2*40e-3;
z = 0:dz:z_max;
new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
new_transducer = new_transducer(1,256:768 -1);
new_transducer = imresize(new_transducer,2,'box');

new_transducer = new_transducer(1,1:N);


index = 1
for z0 = z
%     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
    %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);

    [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
    index = index + 1;
    z0/z_max;
end
I_FSP1_z0 = abs(E_FSP1_z0).^2;

x_min = -15-3;
x_max = 15e-3;
z_min = 10e-3;
z_max = 70e-3;
x = linspace(x_min,x_max,200);
z_field = linspace(z_min,z_max,300);
dz_field = (80e-3 - 10e-3)/300;
index = find(z_field == 40e-3);
line_at_depth_base = create_new_line(delays,amps);
line_at_depth_0 = create_new_line(delays_0,amps);
line_at_depth_1= create_new_line(delays_1,amps);
line_at_depth_gs = create_new_line(delays_gs,amps);
line_at_depth_gs_amps = create_new_line(delays_gs_amps,amps_gs);
line_at_depth_gs_multi_cycle = create_new_line(delays_gs,amps,10);
x_min = -(256 - 1) * pitch/5;
x_max = 256 * pitch/5;
x_for_plot = x_min:pitch/5:x_max;
%x_for_plot = x_full(:,256 - 45:256 + 45);
line_at_depth_base = line_at_depth_base(:,256 - 255:256 + 256);
line_at_depth_0 = line_at_depth_0(:,256 - 255:256 + 256);
line_at_depth_1 = line_at_depth_1(:,256 - 255:256 + 256);
line_at_depth_gs = line_at_depth_gs(:,256 - 255:256 + 256);
line_at_depth_gs_amps = line_at_depth_gs_amps(:,256 - 255:256 + 256);
line_at_depth_gs_multi_cycle = line_at_depth_gs_multi_cycle(:,256 - 255:256 + 256);
source = loaded.source(i,256 - 255:256 + 256);
%shift = find(line_at_depth == 1) - find( line_at_depth_gs == 1);
%line_at_depth = circshift(line_at_depth,-1);
%%
figure
grid on
subplot(2,4,1)
imagesc(x * 1e3,z_field * 1e3,im_from_gs_multi_cycle); colormap jet; axis square; axis on; zoom(1); title('from gs multi cycle', 'FontSize',16)
subplot(2,4,5)
plot(x_for_plot * 1e3,line_at_depth_gs_multi_cycle,'LineWidth',1); title('from gs multi cycle at depth 40 mm','FontSize',16); zoom(1);
subplot(2,4,2)
imagesc(x * 1e3,z_field * 1e3,im_from_calculated); colormap jet; axis square; axis on; zoom(1); title('from gs single cycle', 'FontSize',16)
subplot(2,4,6)
plot(x_for_plot * 1e3,line_at_depth_gs,'LineWidth',1); title('from gs single cycle at depth 40 mm','FontSize',16); zoom(1);
subplot(2,4,3)
imagesc(x * 1e3,z_field * 1e3,im_from_net_0); colormap jet; axis square; axis on; zoom(1); title('from model fine-tuned on integers', 'FontSize',16)
subplot(2,4,7)
plot(x_for_plot * 1e3,line_at_depth_0,'LineWidth',1); title('from model fine-tuned on integers at depth 40 mm','FontSize',16); zoom(1);
subplot(2,4,4)
imagesc(x * 1e3,z_field * 1e3,im_from_net_1); colormap jet; axis square; axis on; zoom(1); title('from model fine-tuned on floats', 'FontSize',16)
subplot(2,4,8)
plot(x_for_plot * 1e3,line_at_depth_1,'LineWidth',1); title('from model fine-tuned on floats at depth 40 mm','FontSize',16); zoom(1);
% subplot(2,3,1)
% plot(x_for_plot,source); title('base pattern');
figure
subplot(2,3,1)
set(gca,'FontSize',20)

plot(x_for_plot * 1e3,line_at_depth_base,'LineWidth',1); title('from net learned on gs at depth 40 mm','FontSize',16); zoom(1);
subplot(2,3,2)
plot(x_for_plot * 1e3,line_at_depth_0,'LineWidth',1); title('from net learned on field ii integers at depth 40 mm','FontSize',16); zoom(1);
subplot(2,3,3)
plot(x_for_plot * 1e3,line_at_depth_1,'LineWidth',1); title('from net learned on field ii at depth 40 mm','FontSize',16); zoom(1);
subplot(2,3,4)
plot(x_for_plot * 1e3,line_at_depth_gs,'LineWidth',1); title('from gs no amps at depth 40 mm','FontSize',16); zoom(1);
subplot(2,3,5)
plot(x_for_plot * 1e3,line_at_depth_gs_amps,'LineWidth',1); title('from gs with amps at depth 40 mm','FontSize',16); zoom(1);

figure
title('from net learned on field ii integers at depth 40 mm','FontSize',16)
plot(x_for_plot * 1e3,line_at_depth_0,DisplayName='model trained on integars');
hold on
plot(x_for_plot * 1e3,line_at_depth_1,DisplayName='model trained on varying focuses');
hold on
plot(x_for_plot * 1e3,line_at_depth_gs,DisplayName='Gershberg-Saxton');
hold on
legend
%%
figure
subplot(1,5,1)
findpeaks(line_at_depth_gs,x_for_plot * 1e3,'Annotate','extents',MinPeakHeight=0.5); title('from gs threshold 0.5')
subplot(1,3,2)
findpeaks(line_at_depth_0,x_for_plot * 1e3,'Annotate','extents',MinPeakHeight=0.5); title('from step 1')
subplot(1,3,3)
findpeaks(line_at_depth_base,x_for_plot * 1e3,'MinPeakProminence',0.01,MinPeakHeight=0.8);
hold on
findpeaks(line_at_depth_0,x_for_plot * 1e3,'MinPeakProminence',0.01,MinPeakHeight=0.8);
[pks,locs,widths,proms] = findpeaks(line_at_depth_base,x_for_plot * 1e3,'MinPeakProminence',0.01,MinPeakHeight=0.8);
[pks,locs,widths_step_1,proms] = findpeaks(line_at_depth_0,x_for_plot * 1e3,'MinPeakProminence',0.01,MinPeakHeight=0.8);
widths
widths_step_1
widths - widths_step_1
mean(widths - widths_step_1)

% figure
% subplot(1,2,1)


% subplot(3,2,1)
% % imagesc(x_half * 1e3,z * 1e3,I_FSP1_z0); colormap jet; axis square; axis on; zoom(1); title('from gs full image')
% imagesc(x_half * 1e3,z * 1e3,im_from_calculated); colormap jet; axis square; axis on; zoom(1); title('from gs full image')
% subplot(3,2,2)
% imagesc(x * 1e3,z_field * 1e3,im_from_field); colormap jet; axis square; axis on; zoom(1); title('from net full image')
% subplot(1,2,1)
% plot(x_full * 1e3,line_at_depth); title('from net at depth 40 mm')
% %plot(x * 1e3,im_from_field(round(40e-3 / dz_field - 10e-3/dz_field),:)); title('from field ii at depth 40 mm')
% subplot(2,2,2)
% plot(x_full * 1e3,line_at_depth_gs); title('from gs at depth 40 mm');
% subplot(2,2,3)
% plot(x_full * 1e3,line_at_depth_gs,'DisplayName','from gs'); title('both plots');
% hold on
% plot(x_full * 1e3,line_at_depth,'DisplayName','from net');
% legend()
%  subplot(3,2,6)
%  findpeaks(line_at_depth,'MinPeakProminence',0.05,'Annotate','extents',MinPeakHeight=0.5); title('peaks from net');
% hold on
% findpeaks(line_at_depth_gs,'MinPeakProminence',0.01,MinPeakHeight=0.3); title('peaks from gs');
% legend()
% [pks,locs,widths,proms] = findpeaks(line_at_depth,x_full * 1e3,'MinPeakProminence',0.05,MinPeakHeight=0.5);
% [pks,locs,widths_gs,proms] = findpeaks(line_at_depth_gs,x_full * 1e3,'MinPeakProminence',0.05,MinPeakHeight=0.5);
% widths
% widths_gs
% mean(widths - widths_gs,'all')
%plot(x_half * 1e3,I_FSP1_z0(40e-3/dz,:)); title('from gs at depth 40 mm');
%%

desired_Output_Shift = generate_patterns(1,1024);
N = 1024;
Transducer = zeros(1,N);
GS = 15;
DZ = 40e-3; % Distance to pattern 
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
pitch = 0.218e-3; % 
Number_of_Elements = 128; % 
Output = desired_Output_Shift;
for n = 1:GS
    Output = (desired_Output_Shift.^0.5).*exp(1i*angle(Output));
%         Output=circshift(Output,[0,3]);
    Transducer = FSP_X_near(Output,-DZ,N,pitch,Wavelength);
%     Transducer = ones(size(Transducer)).*exp(1i*angle(Transducer)); % if you don't allow apodiztion
%         Transducer(1:ceil(N/2)-Number_of_Elements/2) = 0;     % set to zero the pixels outside the transducer
%         Transducer(floor(N/2)+Number_of_Elements/2:end) = 0; % set to zero the pixels outside the transducer
    Transducer(1:512-63) = 0;     % set to zero the pixels outside the transducer
    Transducer(512+64:end) = 0; % set to zero the pixels outside the transducer
    Output = FSP_X_near(Transducer,+DZ,N,pitch,Wavelength);
end
result = Output;

Transducer_Amp = abs(Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2));
Transducer_Amp = Transducer_Amp/max(Transducer_Amp);
Transducer_Phase = angle(Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2));
Transducer_Phase_Reshape = unwrap(Transducer_Phase);
Transducer_Phase_Reshape = Transducer_Phase_Reshape-min(min(Transducer_Phase_Reshape));
Transducer_delay_Wavelength = Transducer_Phase_Reshape/(2*pi);
Transducer_delay_Wavelength = Transducer_delay_Wavelength /Frequancy;



Output = Output/max(Output);
figure
plot(abs(Output),'DisplayName','GS')
hold on
plot(desired_Output_Shift,'DisplayName','base pattern')
hold on
a = Transducer_Amp .* exp(1i * Transducer_Phase_Reshape);
Transducer_temp = zeros(1,N);
Transducer_temp(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2) = a;
res = FSP_X_near(Transducer_temp,+DZ,N,pitch,Wavelength);
res = abs(res) / max(abs(res));
%plot(res,'DisplayName','recovered')

legend
%%
data = table2array(readtable('C:\Users\DrorSchein\Desktop\thesis\thesis\data advanced.csv'));
amp_file = data(:,129:256);
del_file = data(:,257:end);
base = data(:,1:128);
base_padded = padarray(base,[0,(1024 - 128)/2]);
% amp_file = amps;
% del_file = delays;
% base_padded = patterns;
test = zeros(1,N);
pitch = 0.218e-3;
vec_size = 1024;
N = vec_size;
pitch_half = pitch/2;
x_half = -(vec_size - 1)*pitch_half/2:pitch_half:(vec_size)*pitch_half/2;
ii = 50;
e = amp_file(ii,:) .* exp(1j * del_file(ii,:));
x = (-128/2  + 1) *pitch:pitch:(128/2) * pitch;
test(floor(N/2)-Number_of_Elements/2 + 1:floor(N/2)+Number_of_Elements/2 ) =  e;
result = abs(FSP_X_near(test,+DZ,N,pitch,Wavelength));
result = result/max(result);
figure
plot(abs(result))
hold on 
plot(base_padded(ii,:))

%%
%%

a = angle(Transducer_Amp .* exp(1i * Transducer_Phase_Reshape));
b = angle(Transducer_Amp .* exp(1i * Transducer_delay_Wavelength));
c = a - Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2);
d = b - Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2);
figure
plot(a,'DisplayName','reshaped')
hold on
plot(b,'DisplayName','divided')
plot(angle(Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2)),'DisplayName','original')
legend
