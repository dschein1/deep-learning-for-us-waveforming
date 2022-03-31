function total_result = create_new_image(delays,amp)
    arguments
    delays (1,:)
    amp (1,:) = ones(1,128)
    end
    Number_of_cycles=1; % Number of transmitted cycles. 1 for a single pulse
    f0 = 4.464e6; 
    c = 1490; % water in room temperature m/sec (in body  v = 1540)
    pitch = 0.218e-3; % 
    Number_of_Elements = 128; % 
    height = 5e-3; % Height of element [m]
    Fill_Factor = 1;
    kerf = pitch*(1-Fill_Factor); % Kerf [m]
    width = pitch*Fill_Factor; % Width of element
    fs=100e6; %Sampling frequency
    focus = [0 0 40]/1000;
    
    Apo= amp';
    set_sampling(fs);
    set_field('c',c);
    Ts = 1/fs; % Sampling period
    T = Number_of_cycles * 2/f0;
    te = 0:Ts:T; % Time vector
    Th = xdc_linear_array (Number_of_Elements, width, height, kerf, 4, 1, focus);
    impulse_response=sin(2*pi*f0*te+pi);
    impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
    xdc_impulse (Th, impulse_response);
    excitation = sin(2*pi*f0*te+pi); % Excitation signal
    xdc_excitation(Th, excitation);
    xdc_apodization(Th, 0, Apo');
    x_min = -30e-3;
    x_max = 30e-3;
    z_min = 10e-3;
    z_max = 80e-3;
    x = linspace(x_min,x_max,200);
    z = linspace(z_min,z_max,300);
    total_result = zeros(size(delays,1),300,200);
    points = zeros(300,3);
    points(:,3) = z';
    im = zeros(300,200);
    for i=1:size(delays,1)
        xdc_focus_times(Th,0,delays(i,:));
        for j=1:200            
            points(:,1) = x(j);
%     point = [0 0 depth];
            [temp,~] = calc_hp(Th,points);
            p = vecnorm(temp,2,1);
            im(:,j) = p;
        end
        im = im - min(min(im));
        im = im/max(max(im));
        %db_val = 40;
        %const_b = 10^(-db_val/20);
        %const_a = 1-const_b;
        %im = 20*log10(const_a * im +const_b);
        total_result(i,:,:) = im;
    end
%     z_repeat = reshape(repmat(z,1,100),[],1);
%     x_repeat = reshape(repmat(x,100,1),[],1);
%     mat = [x_repeat zeros(length(x_repeat),1) z_repeat];
%     [h,~] = calc_hp(Th,mat);
%     normalized = vecnorm(h,2,1);
%     reshaped = reshape(normalized,100,100);
%     result = abs(reshaped);
%     db_val = 40;
%     result = result - min(min(result));
%     result = result/max(max(result));
%     const_b=10^(-db_val/20);
%     const_a=1-const_b;
%     final = 20*log10(const_a * result + const_b);
%     return final;
end