%%
init_field()
data = readmatrix('datasets/6 focus data delays/base data/0.csv',Range=[2,2,10,776]);
amps = data(:,513:640);
delays = data(:,641:768);
base = data(:,1:512);

combined = amps .* exp(1j * delays);
combined = padarray(combined,[0 448]);
%%
dz = 0.5e-3;
z_max = 2*40e-3;

Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
pitch = 0.218e-3; % 
Number_of_Elements = 128; % 
Transducer_size = pitch*Number_of_Elements;
pitch_half = pitch /2;
z = 0:dz:z_max;
z = flip(z);
for i=1:9
    j = 1;
    for z0 = z
        images_gs(i,j,:) = FSP_X_near(combined(i,:),z0,1024,pitch_half,Wavelength);
        j = j + 1;
    end
end

%%

for i=1:9
    Transducer_Phase = delays(i,:);
    Transducer_Phase_Reshape = unwrap(Transducer_Phase);
    Transducer_Phase_Reshape = Transducer_Phase_Reshape-min(min(Transducer_Phase_Reshape));
    Transducer_Phase_Reshape = Transducer_Phase_Reshape/max(Transducer_Phase_Reshape);
    Transducer_delay_Wavelength = Transducer_Phase_Reshape/(2*pi);
    Transducer_delay_Wavelength = Transducer_delay_Wavelength /Frequancy;
    delays(i,:) = Transducer_delay_Wavelength;
end

for i=1:9
    images_field(i,:,:) = create_new_image(delays(i,:),amps(i,:));
end

%%
x_gs = -(1024-1)*pitch_half/2:pitch_half:1024*pitch_half/2;
x_field = linspace(-15e-3,15e-3,200);
z_field = linspace(10e-3,80e-3,300);
for i=1:9
    figure
    subplot(1,3,1)
    plot(base(i,:))
    subplot(1,3,2)
    to_show(:,:,1) = abs(images_gs(i,:,:));
    imshow(to_show); title('from gs'); colormap hot
    subplot(1,3,3)
    to_show_field(:,:,1) = abs(images_field(i,:,:));
    imshow(to_show_field); title('from field ii'); colormap hot
end
    