base_folder = '.\hydrophone measurments\';
addpath(base_folder);
files = dir('hydrophone measurments\');
base_name = 'single_data';
pitch = 0.218e-3;
N = 1024;
pitch_half = pitch/2;
dz = 0.25e-3;
z_max_gs = 50e-3;
z = 30e-3:dz:50e-3;
z_flipped = flip(z);
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
range_x = round(9.5e-3 / pitch_half);
x_for_image = (-range_x * pitch_half):pitch_half:range_x*pitch_half;
x = -9:0.15:9;

Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
clear E_FSP1_z0;
for index=1:numel(files)
    file_ = files(index);
    if file_.isdir && ~contains(file_.name,'.')
        files_inner = dir([base_folder file_.name]);
        for inner_index=1:numel(files_inner)
            inner_file = files_inner(inner_index);
            inner_file.name
            if contains(inner_file.name,base_name)
                data = load([base_folder file_.name '\' inner_file.name]);
                pattern = double(data.source(1,:));
                patterns = padarray(pattern,[0 256],0,'both');
                [amps, new_transducer] = calculateGS(patterns,false);
                new_transducer = amps .* exp(1i * new_transducer);

                new_transducer = padarray(new_transducer,[0 ((1024 - 128) / 2)]);
                new_transducer = new_transducer(1,256:768 -1);
                new_transducer = imresize(new_transducer,2,'box');
                new_transducer = new_transducer(1,1:N);
                index = 1;
                for z0 = z
                %     [E_FSP1_z0(index,:)] = FSP_X_near(Transducer,z0,N,pitch,Wavelength);
                    %[E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                
                    [E_FSP1_z0(index,:)] = FSP_X_near(new_transducer,z0,N,pitch_half,Wavelength);
                    index = index + 1;
                    z0;
                end
                I_FSP1_z0 = abs(E_FSP1_z0).^2;
                line_gs = abs(FSP_X_near(new_transducer,40e-3,N,pitch_half,Wavelength)).^2;
                line_gs = line_gs(512 - range_x:512 + range_x);
                line_gs = line_gs / max(line_gs);
                I_FSP1_z0 = I_FSP1_z0(:,512 - range_x:512 + range_x);
                figure
                imagesc(x_for_image * 1e3,z * 1e3,I_FSP1_z0); 
                axis equal tight; axis on; colormap hot%jet
                %xlabel('x [mm]', FontSize=16); ylabel('z [mm]',FontSize=16);
                set(gca,'FontSize',20)

                hgexport(gcf, [[base_folder file_.name '\gs image' ]], hgexport('factorystyle'), 'Format', 'tiff');
    
                figure
                plot(x_for_image * 1e3,line_gs);% xlabel('x [mm]', FontSize=16); ylabel('z [mm]',FontSize=16);
                set(gca,'FontSize',20)
                hgexport(gcf, [[base_folder file_.name '\gs line' ]], hgexport('factorystyle'), 'Format', 'tiff');

                index;
            end
            if contains(inner_file.name,'results') && ~contains(inner_file.name,'ppt')
                to_add = 'net';
                if contains(inner_file.name,'gs')
                    to_add = 'gs';
                end
                clear data
                data = load([base_folder file_.name '\' inner_file.name]);
                data=squeeze(data.(['data_' to_add])); 
                clear P
                for m=1:size(data,1)
                     for n=1:size(data,2)
                         if ~isempty(data(m,n).signal)
                %              P(m,n)=max(data(m,n).signal)-min(data(m,n).signal);
                %              P(m,n) = P(m,n)/2;
                             P(m,n)=min(data(m,n).signal);
                             if abs(P(m,n))>15000
                                P(m,n) = 15000;
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
                P(isnan(P)) = 0;
                P = abs(P).^2;
                P = medfilt2(P);
                figure
                imagesc(x,z_flipped * 1e3,P')
                axis tight equal; axis on; colormap hot%jet
                set(gca,'FontSize',20)
                hgexport(gcf, [[base_folder file_.name '\final image' to_add ]], hgexport('factorystyle'), 'Format', 'tiff');
                figure
                line = P(:,round(size(data,2)/2));
                line = line/max(line);
                plot(x,line'); 
                set(gca,'FontSize',20)
                hgexport(gcf, [[base_folder file_.name '\final line' to_add ]], hgexport('factorystyle'), 'Format', 'tiff');
                index;
            end
        end
        close all
    end
end
   close all

