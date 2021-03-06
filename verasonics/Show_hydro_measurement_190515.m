% clc
clear data; clear P
% close all

% x = -L1/2:d1:L1/2;
% z = -L2/2:d2:L2/2;
x = -9:0.1:9;

% data=squeeze(data_SAMI4MHz_204060); z = 18:0.25:66;
% data=squeeze(data_SAMI3MHz_305070); z = 28:0.25:76;
% data=squeeze(data_SAMI4MHz_305070); z = 28:0.25:76;
data=squeeze(data_to_show); z = 36.5:0.15:43.5;
% data=squeeze(data_SAMIinc_305070); z = 28:0.25:76;
% data=squeeze(data_SAMIdec_305070); z = 28:0.25:76;
z = flip(z);
for m=1:size(data,1)
     for n=1:size(data,2)
         if ~isempty(data(m,n).signal)
%              P(m,n)=max(data(m,n).signal)-min(data(m,n).signal);
%              P(m,n) = P(m,n)/2;
             P(m,n)=min(data(m,n).signal);
             if abs(P(m,n))>10000
                P(m,n) = 10000;
             end
%              P(m,n)=max(data(m,n).signal);
%              P(m,n)=sum(abs(data(m,n).signal))/10000;
%              P(m,n)=sqrt(mean(data(m,n).signal.^2));
         else
             P(m,n)=nan;
%              Result(m,n) = nan;
         end
     end
end
%              figure; plot(data(m,n).signal) ; error

P(isnan(P)) = 0;
% P = fillmissing(P,'spline'); % Clean NaN points
% Result = fillmissing(Result,'spline'); % Clean NaN points

% figure
% plot(x,P.^2')
% err

%%
figure
subplot(1,2,1)
imagesc(x,z,P.^2')
axis equal tight; axis on; colormap hot%jet
xlabel('x [mm]'); ylabel('z [mm]'); title('Pressure Field');
set(gca,'FontSize',20)
colorbar
subplot(1,2,2)
line = P(:,round(size(data,2)/2));
plot(x,line'.^2); title('hydrophone measurment at 40 [mm]');
xlabel('x [mm]');
% figure
% for_plot = P(:,27).^2;
% for_plot = for_plot / max(for_plot);
% plot(x,for_plot); title('hydrophone measurment at 40 [mm]');
% loaded = load('C:\Users\Administrator\Documents\MATLAB\Dror\py to matlab\single_data05 focus from net.mat');
% target = loaded.source;
% x_source = -255 * 0.218e-3:0.218e-3:256 * 0.218e-3;
% hold on
% plot(x_source * 1e3,target);
% % err
% %% interp2
% [X,Z] = meshgrid(x,z);
% xq = min(x):(x(2)-x(1))/20:max(x);
% zq = min(z):(z(2)-z(1))/20:max(z);
% 
% [Xq,Zq] = meshgrid(xq,zq);
% 
% Vq = interp2(X,Z,P',Xq,Zq);
% Vq_Norm = Vq - min(min(Vq));
% Vq_Norm = Vq_Norm/max(max(Vq_Norm));
% % Vq_Norm = imrotate(Vq_Norm,0.3);
% 
% Vq_Norm_Scale = Vq_Norm;%.*(Zq/max(max(Zq))).^0.5;
% figure
% imagesc(xq,zq,Vq_Norm_Scale); 
% axis equal tight
% colorbar
% colormap jet        



