clc
clear all
close all
tic
addpath('C:\Users\drors\Desktop\תואר שני\תזה\field 2');
field_init
%%
Show_GS_Process = 0 ;
%% Transducer parameters -> P6-3
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
pitch = 0.218e-3; % 
Number_of_Elements = 128; % 
Transducer_size = pitch*Number_of_Elements;


N=1024; % Vector size for simulations
Zc=N*pitch^2/Wavelength; % Numerical limit (if the z you need is more, increase the N)

x = -(N-1)*pitch/2:pitch:N*pitch/2;
%% GS 
GS = 1e2; % # of GS iterations
DZ = 40e-3; % Distance to pattern 
Diffraction_limit=1.22*Wavelength*DZ/Transducer_size*1e3;
Dx = 5 % in pitchs
Peak_Dis_mm = Dx*pitch*1e3;
Nu_Periods = 1; % Should be odd

desired_Output = zeros(1,N);
desired_Output(floor(N/2)+1-Dx*floor(Nu_Periods/2)+0:Dx:floor(N/2)+1+Dx*floor(Nu_Periods/2)+0) = 1;
% f0=1/(Dx*pitch);
% desired_Output=0.5+0.5*sin(2*pi*x*f0);
% desired_Output(1:round(N/2)-(Nu_Periods*Dx)/2) = 0;     % set to zero the pixels outside the transducer
% desired_Output(round(N/2)+(Nu_Periods*Dx)/2+1:end) = 0; % set to zero the pixels outside the transducer
desired_Output = generate_patterns(1,1024,8);
desired_Output = desired_Output(1,:);

% desired_Output(501)=1;
%%
for Shift = 0
    desired_Output_Shift = circshift(desired_Output,[0,Shift]);
    Transducer = zeros(1,N);
    Output = desired_Output_Shift;
    RMS = zeros(1,GS);
    if Show_GS_Process == 1
        figure
    end
    for n = 1:GS
        n
        Output = (desired_Output_Shift.^0.5).*exp(1i*angle(Output));
%         Output=circshift(Output,[0,3]);
        Transducer = FSP_X_near(Output,-DZ,N,pitch,Wavelength);
         Transducer = ones(size(Transducer)).*exp(1i*angle(Transducer)); % if you don't allow apodiztion
%         Transducer(1:ceil(N/2)-Number_of_Elements/2) = 0;     % set to zero the pixels outside the transducer
%         Transducer(floor(N/2)+Number_of_Elements/2:end) = 0; % set to zero the pixels outside the transducer
        Transducer(1:501-63) = 0;     % set to zero the pixels outside the transducer
        Transducer(501+64:end) = 0; % set to zero the pixels outside the transducer
        Output = FSP_X_near(Transducer,+DZ,N,pitch,Wavelength);

    end



%     % GS Results
%     figure
%     set(gcf, 'Position', get(0,'Screensize')); 
%     subplot(3,2,1)
%     bar(x*1e3,abs(Transducer)); title('Transducer Abs'); xlim([-Transducer_size*1e3,+Transducer_size*1e3]); xlabel('x[mm]');
%     subplot(3,2,2)
%     plot(x*1e3,angle(Transducer)); title('Transducer Phase'); xlim([-Transducer_size*1e3,+Transducer_size*1e3]); xlabel('x[mm]');
%     subplot(3,2,3)
%     bar(x*1e3,desired_Output_Shift); title('Desired Output Abs'); xlim([-10,+10]); xlabel('x[mm]'); %ylim([0,1.1]);
%     subplot(3,2,5)
%     bar(x*1e3,abs(Output).^2); title('Output Abs'); xlim([-10,10]); xlabel('x[mm]');
%     subplot(3,2,6)
%     plot(x*1e3,angle(Output)); title('Output Phase'); xlim([-Transducer_size*1e3,+Transducer_size*1e3]); xlabel('x[mm]');
%     subplot(3,2,4)
%     I_GS_Output = abs(Output(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2)).^2;
% 



end
% error
%% FSP
% Transducer_2pix=circshift(Transducer,[0,-500]);
Transducer_2pix = Transducer(256:768 - 1);
Transducer_2pix = imresize(Transducer_2pix,2,'box');
Transducer_2pix = Transducer_2pix(1,1:N);
%%
[amps,delays] = calculateGS(desired_Output,false);
new_transducer = amps .* exp(1i * delays);
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
dz = 0.1e-3;
z_max = 2*40e-3;
z = 0:dz:z_max;
new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
new_transducer = new_transducer(1,256:768 -1);
new_transducer = imresize(new_transducer,2,'box');

new_transducer = new_transducer(1,1:N);
figure
subplot(2,2,1)
plot(abs(Transducer_2pix) .^ 2); title('abs from source');
subplot(2,2,2)
plot(angle(Transducer_2pix)); title('angle from source');
subplot(2,2,3)
plot(abs(new_transducer).^ 2); title('abs from function');
subplot(2,2,4)
plot(angle(new_transducer)); title('angle from function')



index = 1
for z0 = z
%     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
    %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);

    [E_FSP1_z0(index,:)] = FSP_X_near(Transducer_2pix,z0,N,pitch_half,Wavelength);
    index = index + 1;
    z0/z_max;
end

I_FSP1_z0 = abs(E_FSP1_z0).^2;
%I_FSP1_z0 = I_FSP1_z0/max(max(I_FSP1_z0));
% 
% figure
% subplot(1,2,1)
% imagesc(x_half*1e3,z*1e3,abs(E_FSP1_z0).^2); axis square; axis on; title('|Abs|^2'); colormap jet; zoom(1);
% xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
% subplot(1,2,2)
% imagesc(x_half*1e3,z*1e3,real(E_FSP1_z0)); axis square; axis on; title('Real'); colormap jet; zoom(1);
% xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');

% error


figure
subplot(1,2,2)
set(gcf, 'Position',[0,0,1000,1000]);
imagesc(x_half*1e3,z*1e3,I_FSP1_z0); axis square; axis on; title('Intensity [a.u.]'); colormap jet; zoom(1);
xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
set(gca,'FontSize',20)
subplot(1,2,1)
res = I_FSP1_z0(DZ/dz,:);
x = linspace(-15e-3,15e-3,200);
plot(x_half,res)
%figure
%plot(x_half*1e3,I_FSP1_z0(30/0.5,:));  axis square; axis on; title('Intensity [a.u.]'); colormap jet; zoom(1);
%plot(x_half*1e3,desired_Output);  axis square; axis on; title('Intensity [a.u.]'); colormap jet; zoom(1);
%xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
%set(gca,'FontSize',20)
% figure
% thre = prctile(I_FSP1_z0,99.5,[1 2])
% h = imbinarize(I_FSP1_z0,thre);
% % figure
% % set(gcf, 'Position',[0,0,1000,1000]);
% % imagesc(x_half*1e3,z*1e3,h); axis square; axis on; title('Intensity [a.u.]'); colormap gray; zoom(1);
% % xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
% % set(gca,'FontSize',20)
% final = zeros(size(h));
% s = regionprops('table',h,'Centroid');
% centroids = round(cat(1,s.Centroid));
% indexs = sub2ind(size(final),centroids(:,2),centroids(:,1));
% final(indexs) = 0.8;
% %plot(centroids(:,1),centroids(:,2),'b*')
% % figure
% %set(gcf, 'Position',[0,0,1000,1000]);
% imagesc(x_half*1e3,z*1e3,final); axis square; axis on; title('Intensity [a.u.]'); colormap gray; zoom(1);
% xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); xlabel('x[mm]'); ylabel('z[mm]');
% set(gca,'FontSize',20)
%%
figure
joined = cat(2,I_FSP1_z0(:,300:750),h(:,300:750),final(:,300:750));
imshow(joined); colormap gray;
%%
% az = 45;
% el = 40;
% figure
% set(gcf, 'Position',[0,0,1600,1000]);
% surf(x_half*1e3,z*1e3,I_FSP1_z0); xlabel('x[mm]'); ylabel('z[mm]'); zlabel('Intensity [a.u.]'); colormap jet ;
% xlim([-Transducer_size*1e3/2,+Transducer_size*1e3/2]); view(az,el)
% set(gca,'FontSize',20)


%%
Transducer_Amp = abs(Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2));
Transducer_Amp = Transducer_Amp/max(Transducer_Amp);
Transducer_Phase = angle(Transducer(floor(N/2)-Number_of_Elements/2+1:floor(N/2)+Number_of_Elements/2));
Transducer_Phase_Reshape = unwrap(Transducer_Phase);
Transducer_Phase_Reshape = Transducer_Phase_Reshape-min(min(Transducer_Phase_Reshape));
Transducer_delay_Wavelength = Transducer_Phase_Reshape/(2*pi);
Transducer_delay_Wavelength = Transducer_delay_Wavelength / Frequancy;

generated = create_new_line(Transducer_delay_Wavelength,Transducer_Amp);
im = squeeze(create_new_image(Transducer_delay_Wavelength,Transducer_Amp));
x = linspace(-15e-3,15e-3,200);
dz = 0.5e-3;
z_max = 2*40e-3;
z = 0:dz:z_max;
z = flip(z);
[amp_func, delay_func] = calculateGS(desired_Output);
gen_func = create_new_line(delay_func,amp_func);
figure
subplot(2,2,1)
plot(x,generated)
subplot(2,2,4)
plot(x,gen_func)
subplot(2,2,2)
imagesc(x*1e3,flip(z)*1e3,im); title('Intensity [a.u.]'); colormap jet;
% figure
% subplot(1,4,1)
% plot(Transducer_Amp); title('Transducer Abs'); xlim([1,Number_of_Elements]); ylim([0,1]); ylabel('Abs');
% subplot(1,4,2)
% plot(Transducer_Phase); title('Transducer Phase'); xlim([1,Number_of_Elements]); ylabel('Phase');
% subplot(1,4,3)
% plot(Transducer_Phase_Reshape); title('Transducer Phase'); xlim([1,Number_of_Elements]); ylabel('Unwrap Phase');
% subplot(1,4,4)
% plot(Transducer_delay_Wavelength); title('Transducer Delay'); xlim([1,Number_of_Elements]); ylabel('[\lambda]');
%% Save

File_Export(1,:) = Transducer_Amp;
File_Export(2,:) = Transducer_delay_Wavelength;

% save([num2str(Nu_Periods),'peaks@',num2str(DZ*1e3),'mm ',num2str(Peak_Dis_mm),'mm spacing Shift#0.txt'],'File_Export','-ascii','-tabs');

% Test_File = load([num2str(Nu_Periods),'peaks@',num2str(DZ*1e3),'mm ',num2str(Peak_Dis_mm),'mm spacing Shift#-1.txt'])

% %% Compare to geomtrical modal
% xx = (-32:31)*pitch;
% dxx = DZ-(xx.^2+DZ^2).^0.5;
% % Dt_ns = dxx/v*1e9;
% % Dt_ns = Dt_ns-min(Dt_ns);
% D_Lambda = dxx/Wavelength;
% D_Lambda = D_Lambda-min(D_Lambda);
% 
% figure
% % plot(xx,Transducer_delay_ns,xx,Dt_ns); legend('Transducer Delay','Geometrical model'); ylabel('[ns]');
% plot(xx,Transducer_delay_Wavelength,xx,D_Lambda); legend('Transducer Delay','Geometrical model'); ylabel('[\lambda]');
