init_field()
%%
batch_size = 5000;
vec_size = 1024;
N = vec_size;
DZ = 40e-3; % Distance to pattern 
Frequancy = 4.464e6; 
v = 1490; % water in room temperature m/sec (in body  v = 1540)
Wavelength = v/Frequancy;
pitch = 0.218e-3; % 
amps = zeros(batch_size,128);
delays = zeros(batch_size,128);

patterns = generate_patterns(batch_size,vec_size);
results = zeros(batch_size,256);
for i=1:batch_size
    [amps(i,:) , delays(i,:)] = calculateGS(patterns(i,:));
    %results(i,:) = calculateGS(patterns(i,:));
    %results(i,:) = abs(results(i,:));
end


%%
Number_of_Elements = 128;
%current = results(:,512-128 + 1:512+128);
j = 20;
e = amps(j,:) .* exp(1i * delays(j,:));
%x = -N/2 *pitch:pitch:(N/2 - 1) * pitch;
test = zeros(1,N);
test(floor(N/2)-Number_of_Elements/2 + 1:floor(N/2)+Number_of_Elements/2) =  e;

result = abs(FSP_X_near(test,+DZ,N,pitch,Wavelength));
result = result / max(result);
figure
plot(result)
hold on
plot(patterns(j,:))
%%
figure
subplot(2,2,1)
pitch_half = pitch/2;
x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
plot(x_half,abs(results(1,:)).^2)
subplot(2,2,2)
plot(x_half,abs(results(2,:)).^2)
subplot(2,2,3)
plot(x_half,abs(results(3,:)).^2)
subplot(2,2,4)
plot(x_half,abs(results(4,:)).^2)
%%
full = [patterns results];
%%
writematrix(full,'C:\Users\drors\Desktop\code for thesis\data gs.csv')

%%

ele = 1:128;
x = linspace(-15e-3,15e3,200);
dz = 0.5e-3;
z_max = 2*40e-3;
z = 0:dz:z_max;
z = flip(z);
pattern = zeros(1,1001);
pattern(1,501) = 1;
[amps, delays] = calculateGS(pattern);
delays_calc = calc_delay(128,1490,[0 0 40e-3],0.218e-3);
figure
% norm(delays)
% norm(delays_calc)
% plot(ele, delays)
% hold on
% plot(ele,delays_calc)
% legend('from GS','calculated')
line_from_GS = create_new_line(delays,amps);
line_from_calculated = create_new_line(delays_calc);
subplot(1,3,1)
plot(x,line_from_GS)
hold on
plot(x,line_from_calculated)
legend('from GS','calculated')
subplot(1,3,2)
im_calc = squeeze(create_new_image(delays_calc));
im_gs = squeeze(create_new_image(delays,amps));
imagesc(x*1e3,flip(z)*1e3,im_calc); title('Intensity [a.u.]'); colormap jet;
subplot(1,3,3)
imagesc(x*1e3,flip(z)*1e3,im_calc); title('Intensity [a.u.]'); colormap jet;
%%
batch_size = 20;
vec_size = 1001;
patterns = generate_patterns(batch_size,vec_size);
figure
for i=1:batch_size
    subplot(4,5,i)
    x_pattern_center = linspace(-15e-3,15e-3,128);
    plot(x_pattern_center,patterns(i,500-128/2 + 1:500+128/2))
end
%%
batch_size = 5000;
vec_size = 1001;
N = vec_size;
patterns = generate_patterns(batch_size,vec_size);
% [row,col] = find(patterns);
% [~,order] = sort(row);
% col = col(order);
% points =  (col - 500) * pitch
amps = zeros(batch_size,128);
pitch = 0.218e-3;
pitch_half = pitch/2;
x_half = -(vec_size-1)*pitch_half/2:pitch_half:vec_size*pitch_half/2;
dz = 0.5e-3;
z_max = 2*40e-3;
z = 0:dz:z_max;
z = flip(z);
x = linspace(-15e-3,15e-3,200);
pitch = 0.218e-3; % 
close all
x_pattern = linspace(-500 * pitch,500*pitch,1001);
x_pattern_center = linspace(-15e-3,15e-3,128);
delays = zeros(batch_size,128);
results = zeros(batch_size,200);
Dx = 5; % in pitchs
Peak_Dis_mm = Dx*pitch*1e3;
Nu_Periods = 11; % Should be odd
patterns(1,:) = 0;
patterns(1,floor(N/2)+1-Dx*floor(Nu_Periods/2)+0:Dx:floor(N/2)+1+Dx*floor(Nu_Periods/2)+0) = 1;

for i=1:batch_size
    [amps(i,:) , delays(i,:)] = calculateGS(patterns(i,:));
    results(i,:) = create_new_line(delays(i,:));
    %im = squeeze(create_new_image(delays(i,:)));
%     delay = calc_delay(128,1490,[points(i) 0 40e-3],pitch);
%     result = create_new_line(delay);
%     calc_im = squeeze(create_new_image(delay));
%     figure;
%     subplot(1,5,2)
%     plot(x,results(i,:))    
%     title('at image plane')
%     subplot(1,5,5)
%     plot(ele,amps(i,:))
%     title('amplitudes')
%     subplot(1,5,4)
%     plot(ele,delays(i,:))
%     title('delays')
%     subplot(1,5,1)
%     plot(x_pattern_center,patterns(i,500-128/2 + 1:500+128/2))
%     title('base pattern')
    %subplot(1,5,3)
    %imagesc(x*1e3,flip(z)*1e3,im); title('Intensity [a.u.]'); colormap jet;
    
%     subplot(1,5,2)
%     plot(x,result)    
%     title('at image plane - calc')
%     subplot(1,5,3)
%     imagesc(x*1e3,flip(z)*1e3,calc_im); title('Intensity [a.u.] - calc'); colormap jet;
end

%%
% full = [results delays amps];
full = [patterns(:,512-128/2 + 1:512+128/2) amps delays];
%%
writematrix(full,'C:\Users\DrorSchein\Desktop\thesis\thesis\data advanced.csv')

