

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
data -
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
