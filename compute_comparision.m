function [total_mse_conv, total_results, total_amounts] = compute_comparision( ...
    source_patterns, ...
    convolved_patterns, ...
    delays_net_0, ...
    delays_net_1, ...
    cycles, ...
    depth, ...
    method, ...
    file_path, ...
    save_examples)
    pitch = 0.218e-3;
    x_min = -(256) * pitch/3;
    x_max = (256 - 1) * pitch/3;
    x_full = x_min:pitch/3:x_max;
    x_min = -17e-3;
    x_max = 17e-3;
    z_min = 20e-3;
    z_max = 60e-3;
    x = linspace(x_min,x_max,200);
    N = 1024;
    pitch_half = pitch/2;
    x_half = -(N-1)*pitch_half/2:pitch_half:N*pitch_half/2;
    x_full_gs = -(512-1)*pitch/2:pitch:512*pitch/2;
    x_lowest_gs = x_full_gs(1);
    x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
    range_x = round(17e-3 / pitch_half);
    dz = 0.1e-3;
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    mode = method;
    total_size = size(delays_net_0, 1);
    size_of_step = floor(total_size / 3);
    peak_height = 0.5;
    
    total_results = zeros(3,10,3);
    total_amounts = zeros(10,3);
    total_mse_conv = zeros(3,10,3);
    for i=1:3
        success_gs = zeros(1,10)';
        mse_gs_conv = zeros(1,10)';
        success_0 = zeros(1,10)';
        mse_0_conv = zeros(1,10)';
        mse_1_conv = zeros(1,10)';
        success_1 = zeros(1,10)';
        num_for_each = zeros(10,1);
        for j=(i-1) * size_of_step + 1:i *size_of_step
            base_pattern = source_patterns(j,:);
            patterns = padarray(base_pattern,[0 256],0,'both');
            [amps, delays_gs] = calculateGS(patterns,false, 40e-3);
            delays_gs = normalize_delays(delays_gs);
            delays_0 = normalize_delays(delays_net_0(j,:));
            delays_1 = normalize_delays(delays_net_1(j,:));
            line_at_depth_gs = create_new_line(delays_gs,ones(1,128),cycles,3,depth);
            line_at_depth_0 = create_new_line(delays_0,ones(1,128),cycles,3,depth);
            line_at_depth_1 = create_new_line(delays_1,ones(1,128),cycles,3,depth);
            [pks_base,locs_base,widths_base ,~] = findpeaks(base_pattern,x_full_gs,'MinPeakHeight',peak_height);
            locs_base = round(locs_base / (pitch/3));
            locs_base = locs_base + 256;
            pks_base = pks_base (locs_base > 0);
            locs_base = locs_base (locs_base > 0);
            pattern_for_comparision = zeros(1,512);
            pattern_for_comparision(locs_base) = pks_base;
            pattern_for_comparision_conv = interp1(x_full_gs,convolved_patterns(j,:),x_full);
            if strcmp(mode,'floats')
                [pks_gs,locs_gs] = extract_peaks(line_at_depth_gs,pattern_for_comparision);
                [pks_0,locs_0] = extract_peaks(line_at_depth_0, pattern_for_comparision);
            else
                [pks_gs,locs_gs,widths_base_gs,~] = findpeaks(line_at_depth_gs,x_full,'MinPeakHeight',peak_height);
                [pks_0,locs_0,widths_base_0,~] = findpeaks(line_at_depth_0,x_full,'MinPeakHeight',peak_height);
            end

            num_peaks = length(pks_base);
            if num_peaks < 11 && num_peaks > 0
                num_for_each(num_peaks) = num_for_each(num_peaks) + 1;
                if length(pks_gs) == num_peaks
                    before = success_gs(num_peaks);
                    success_gs(num_peaks) =  before + 1;
                end
                if length(pks_0) == num_peaks
                    before = success_0(num_peaks);
                    success_0(num_peaks) = before + 1;
                end
                if save_examples == true
                    example_path = append(file_path, int2str(j), ' ', int2str(num_peaks), '.tif');
                    save_example_to_file(patterns, delays_gs, delays_0, example_path);
                end
                mse_gs_conv(num_peaks) = mse_gs_conv(num_peaks) + sum((line_at_depth_gs - pattern_for_comparision_conv).^2);
                mse_0_conv(num_peaks) = mse_0_conv(num_peaks) + sum((line_at_depth_0 - pattern_for_comparision_conv).^2);
            end
    
        end
            total_results(1,:,i) = success_gs';
            total_results(2,:,i) = success_0';
            total_results(3,:,i) = success_1'; 
            total_amounts(:,i) = num_for_each;
            total_mse_conv(1,:,i) = mse_gs_conv';
            total_mse_conv(2,:,i) = mse_0_conv';
            total_mse_conv(3,:,i) = mse_1_conv';
    end
end




function [peaks,locs] = extract_peaks(input, pattern)
    pitch = 0.218e-3;
    x_min = -(256 - 1) * pitch/3;
    x_max = 256 * pitch/3;
    x_full = x_min:pitch/3:x_max;
    peaks = [];
    locs = [];
    peak_height = 0.5;
    [pks_base,locs_base,~,~] = findpeaks(pattern,'MinPeakHeight',peak_height);
    [pks_input,locs_input,~,~] = findpeaks(input,'MinPeakHeight',peak_height/2,'MinPeakDistance',2);
    j = 1;
    for i=1:length(pks_input)
        if j > length(pks_base) && pks_input(i) > peak_height
            peaks = [];
            locs = [];
            break
        elseif j > length(pks_base)
            continue
        end
        if  pks_base(j)/2 < pks_input(i) && abs(locs_base(j) - locs_input(i)) < 30
            peaks = [peaks pks_input(i)];
            locs = [locs locs_input(i)];
            j = j + 1;
        elseif pks_input(i) > peak_height
            peaks = [];
            locs = [];
            break
        end
    end
end