close all
clear all
clc

addpath('C:\Users\drors\Desktop\תואר שני\תזה\field 2');
field_init
close
% Parameters definition

Number_of_cycles=1; % Number of transmitted cycles. 1 for a single pulse
f0 =4.464e6; % transmission frequency
c = 1490; % water in room temperature m/sec (in body  v = 1540)
Lambda = c/f0;
Pitch = 0.218e-3; % Element size
Number_of_Elements = 128; % %Number of elements (128 for P6-3 ATL)
Transducer_size = Pitch*Number_of_Elements;
fs=100e6; %Sampling frequency
rho = 1000; %1000 kg/m^3 or 1g/cm^3
% Vec=Pitch*[1:Number_of_Elements];
% Vec=Vec-max(Vec)/2;
Fill_Factor = 1;
width = Pitch*Fill_Factor; % Width of element
height = 5e-3; % Height of element [m]
kerf = Pitch*(1-Fill_Factor); % Kerf [m]
elefocus = 1; % Whether to use elevation focus
Rfocus = 60e-3; % Elevation focus [m]
spacing= width + kerf;


    %% Calculating the emitted pressure field

        % This sets up our coordinate grid to cover the full width of the 
        % transducer array in the x direction and to measure the pressure field to
        % zmax[mm] in the z direction. 
        Pitch_half = Pitch/2;
        N=500; % Vector size for simulations
        xmin = -15e-3;%-(N-1)*Pitch_half/2;%-(width + spacing) * (Number_of_Elements/2+1);
        xmax = 15e-3;%N*Pitch_half/2;%(width + spacing) * (Number_of_Elements/2+1);
        zmin = 10e-3; %for FieldII zmim can't be 0. Min=1mm
        zmax = 50e-3;
        dx = 0.1e-3;%Pitch_half;%(xmax-xmin)/xpoints;
        dz  = 0.2e-3;%(zmax-zmin)/zpoints;
        x = xmin:dx:xmax;
        z = zmin:dz:zmax;

        Apo= ones(Number_of_Elements,1);
        focus = [10 0 40]/1000; % Fixed focal point [m]

        % This sets the sampling frequency, and the excitation signal of the
        % apertures.
        set_sampling(fs);
        set_field('c',c);
        Ts = 1/fs; % Sampling period
        T = Number_of_cycles*2/f0;
        te = 0:Ts:T; % Time vector

        % Define the transducer
        Th = xdc_linear_array (Number_of_Elements, width, height, kerf, 4, 1, focus);
%         xdc_apodization(Th, 0, Apo);
        %xdc_focus_times(Th,0,Delay*Lambda/(2*pi*c));%*Lambda/c


        impulse_response=sin(2*pi*f0*te+pi);
        impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
        xdc_impulse (Th, impulse_response);
        excitation = sin(2*pi*f0*te+pi); % Excitation signal
        xdc_excitation(Th, excitation);
%         num_focuses = randi(3);
%         points = zeros(num_focuses,3);
%         points(:,3) = 40e-3;
%         for j=1:num_focuses
%             points(j,1) = x(randi([round((j-1)*200/num_focuses)+1 round(j*200/num_focuses)]));
%         end
%         point = [0 0 40e-3]; 
%         delays = calc_delay(Number_of_Elements,c,point,Pitch);
%         xdc_focus_times(Th,0,delays);
        % The next step is to initialize the vectors, run the calc_hp function
        % and display the resulting pressure field.
        tic;
        point = [0 0 0];
        t1 = zeros(length(x),length(z)); % Start time for Thrture 1 pressure signal
        P1n = t1; % Norm of pressure from Thrture 1
        for n2 = 1:length(x)
            clc;[ n2 length(x)]
            for n3 = 1:length(z)
                point(1)=x(n2);
                point(2)=0;
                point(3)=z(n3);
                [p1,t1(n2,n3)] = calc_hp(Th,point);
                P1n(n2,n3) = norm(p1);
            end
        end
        %
        I_FSP1_z0=squeeze(abs(P1n));
        I_FSP1_z0=rot90(I_FSP1_z0,1);
        I_FSP1_z0=flipud(I_FSP1_z0);
        Single_Focus_PSF=I_FSP1_z0;

    dB_val=20; 
    
    Single_Focus_PSF=Single_Focus_PSF-min(min(Single_Focus_PSF));
    Single_Focus_PSF=Single_Focus_PSF/max(max(Single_Focus_PSF));
    Const_b=10^(-dB_val/20);
    Const_a=1-Const_b;
    Single_Focus_PSF_dB=20*log10(Const_a*Single_Focus_PSF+Const_b);
    
    %%
    
    depth = 40e-3;
    new_dz = (80e-3 - 10e-3)/300;
    dx = (15e-3 + 15e-3)/200;
    index = round((depth - 10e-3)/new_dz);
    x_min = -15e-3;
    x_max = 15e-3;
    x = linspace(x_min,x_max,200);
    D = 0.218e-3 * 128;
    minimal_distance = (1.206 * (c/f0) * 40e-3) / D;
    minimum = minimal_distance / dx;
%%
    full = zeros(50,328);
    focuses = zeros(50,3);
    for i=0:50
        new_focus = [x(min(round((i+1)*length(x)/50),200)) 0 40e-3];
        focuses(i+1,:) = new_focus;
        delays = calc_delay(Number_of_Elements,c,new_focus,Pitch)';
        xdc_focus_times(Th,0,delays);
        single = generate_pressure_line(Th,depth); 
        full(i+1,:) = [single delays];
    end
%     for i=51:300
%         num_focuses = randi(3);
%         points = zeros(num_focuses,3);
%         points(:,3) = 40e-3;
%         for j=1:num_focuses
%             points(j,1) = randi([round((j-1)*200/num_focuses)+1 round(j*200/num_focuses)]);
%         end
%         delays = calc_delay(Number_of_Elements,c,points,Pitch)';
%         xdc_focus_times(Th,0,delays');
%         single = generate_pressure_line(Th,depth); 
%         full(i+1,:) = [single delays'];
%     end
    writematrix(full,'C:\Users\drors\Desktop\code for thesis\data original.csv') 
%%
        num_focuses = randi(3);
        points = zeros(num_focuses,3);
        points(:,3) = 40e-3;
        for j=1:num_focuses
            points(j,1) = x(randi([round((j-1)*200/num_focuses)+1 round(j*200/num_focuses)]));
        end
        
        point = [0 0 depth];
        delays = calc_delay(Number_of_Elements,c,point,Pitch)';
        xdc_focus_times(Th,0,delays);
        single = generate_pressure_line(Th,depth); 
        %plot(x,single)
        point
        ele = -128/2+1:128/2;
        plot(ele,delays)
    %%


        figure()
        imagesc(x*1e3,z*1e3,Single_Focus_PSF); axis image; axis on; colormap hot;%caxis([-dB_val, 0])%caxis([Min_val, Max_val])
%         xlim ([-10 10])
        xlabel('x[mm]'); ylabel('z[mm]');
%       ylim([15,100]);
        set(gca,'Fontsize',20); 
%         hgexport(gcf, ['Delay_focused_to_30_angle.tif'], hgexport('factorystyle'), 'Format', 'tiff');
%         saveas(h2,'PSF_EDOF.tif')
%%
thre = prctile(I_FSP1_z0,99.5,[1 2])
h = imbinarize(I_FSP1_z0,thre);
figure
imagesc(x*1e3,z*1e3,h); axis square; axis on; title('Intensity [a.u.]'); colormap gray;
%xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); 
xlabel('x[mm]'); ylabel('z[mm]');
set(gca,'FontSize',20)

final = zeros(size(h));
s = regionprops('table',h,'Centroid');
centroids = round(cat(1,s.Centroid));
indexs = sub2ind(size(final),centroids(:,2),centroids(:,1));
final(indexs) = 1;
%plot(centroids(:,1),centroids(:,2),'b*')
figure
%set(gcf, 'Position',[0,0,1000,1000]);
imagesc(x*1e3,z*1e3,final); axis square; axis on; title('Intensity [a.u.]'); colormap gray; zoom(1);
xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
set(gca,'FontSize',20)

figure
joined = cat(2,Single_Focus_PSF,h,final);
imshow(joined); colormap hot;
    %% Save the data
%     Transducer_Amp = Apo;%abs(Transducer(N/2-Number_of_Elements/2+1:N/2+Number_of_Elements/2));
%     Transducer_Amp = Transducer_Amp/max(Transducer_Amp);
%     % Transducer_Phase = angle(Transducer(N/2-Number_of_Elements/2+1:N/2+Number_of_Elements/2));
%     Transducer_Phase_Reshape = Delay;%unwrap(Transducer_Phase);
%     Transducer_Phase_Reshape = Transducer_Phase_Reshape-min(min(Transducer_Phase_Reshape));
%     Transducer_delay_Wavelength = Transducer_Phase_Reshape/(2*pi);

    %% Save

%     File_Export(1,:) = Transducer_Amp;
%     File_Export(2,:) = Transducer_delay_Wavelength;
