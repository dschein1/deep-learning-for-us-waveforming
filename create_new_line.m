function total_result = create_new_line(delays,	amp)
    arguments
    delays (1,:)
    amp (1,:) = ones(1,128)
    end
    depth = 35e-3;
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
    focus = [0 0 46]/1000;
    
    Apo= amp';
    set_sampling(fs);
    set_field('c',c);
    Ts = 1/fs; % Sampling period
    T = Number_of_cycles*2/f0;
    te = 0:Ts:T; % Time vector
    Th = xdc_linear_array (Number_of_Elements, width, height, kerf, 4, 1, focus);
    impulse_response=sin(2*pi*f0*te+pi);
    impulse_response=impulse_response.*hanning(max(size(impulse_response)))';
    xdc_impulse (Th, impulse_response);
    excitation = sin(2*pi*f0*te+pi); % Excitation signal
    xdc_excitation(Th, excitation);
    xdc_apodization(Th, 0, Apo');
    total_result = zeros(size(delays,1),200);
    %%generating the actual line
    for i=1:size(delays,1)
        xdc_focus_times(Th,0,delays(i,:));
        x_min = -15e-3;
        x_max = 15e-3;
        x = linspace(x_min,x_max,200);
        points = zeros(200,3);
        points(:,1) = x';
        points(:,3) = depth;
%     point = [0 0 depth];
        [temp,~] = calc_hp(Th,points);
        p = vecnorm(temp,2,1);
        p = abs(p);
        p = p - min(min(p));
        total_result(i,:) = p/max(max(p)) ;
    end
%     
%     for ii=1:length(x)
%         clc; ii
%         point(1)=x(ii);
%         [temp,~] = calc_hp(Th,point);
%         p(ii) = norm(temp); 
%     end