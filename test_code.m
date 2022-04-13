init_field

pitch = 0.218e-3;
delays_calc = calc_delay(128,pitch,1490,[0 0 40]/1000); 
pattern= zeros(1,1024);
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
patterns = zeros(1,1024);
%patterns(490) = 1;
%patterns(500) = 1;
patterns(512 - 20) = 1;
patterns(512 + 20) = 1;
patterns(512) = 1;
patterns(512 + 10) = 1;
patterns(512 - 10) = 1;
[amps, delays_gs] = calculateGS(patterns,false);
loaded = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\curriculum 1.mat');
delays = double(loaded.from_net(10,:));
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
figure
subplot(1,2,1)
plot(delays_gs); title('delays from gs');
subplot(1,2,2)
 plot(delays); title('delays from model');
%  delays = delays - min(min(delays));
% delays = unwrap(delays);
% delays = delays / (2*pi);
% delays = delays / Frequancy;
 
% subplot(1,3,3)
% plot(delays ./ delays_calc); title('delays from GS divided by delays from calculation')


im_from_calculated = squeeze(create_new_image(delays_calc));
im_from_field = squeeze(create_new_image(delays,amps));
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

x_min = -30e-3;
x_max = 30e-3;
z_min = 10e-3;
z_max = 80e-3;
x = linspace(x_min,x_max,200);
z_field = linspace(z_min,z_max,300);
dz_field = (80e-3 - 10e-3)/300;
index = find(z_field == 40e-3);
figure
subplot(2,2,1)
imagesc(x_half * 1e3,z * 1e3,I_FSP1_z0); colormap jet; axis square; axis on; zoom(1); title('from gs full image')
subplot(2,2,2)
imagesc(x * 1e3,z_field * 1e3,im_from_field); colormap jet; axis square; axis on; zoom(1); title('from field ii full image')
subplot(2,2,4)
plot(x * 1e3,im_from_field(round(40e-3 / dz_field - 10e-3/dz_field),:)); title('from field ii at depth 40 mm')
subplot(2,2,3)
plot(x_half * 1e3,I_FSP1_z0(40e-3/dz,:)); title('from gs at depth 40 mm');
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
