function compare_methods(method, cycles, save_examples, depth)
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
    x_for_line = x_full_gs(length(x_full_gs)/2 - 200:length(x_full_gs)/2 + 200);
    range_x = round(17e-3 / pitch_half);
    dz = 0.1e-3;
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    mode = method;
    if strcmp(mode,'floats')
    %loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision uniform.mat');
        loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision.mat');
    else
    %loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparision uniform integers.mat');
        loaded_for_comparision = load('C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab\batch for comparisionintegers.mat');
    end
    if save_examples == true
        folder_path = append('results/',num2str(cycles),' cycles/');
        mkdir(folder_path);
    end
    delays_base = loaded_for_comparision.from_gs;
    source = loaded_for_comparision.source;
    x_conv = loaded_for_comparision.x_acc;
    delays_net_base = loaded_for_comparision.from_net_base;
    delays_step_0 = loaded_for_comparision.from_net_step_0;
    delays_step_1 = loaded_for_comparision.from_net_step_1;
    [total_mse_conv, total_results, total_amounts] = compute_comparision(source, x_conv, delays_step_0, delays_step_1, cycles, depth, method);
    full.total_results = total_results;
    full.total_amount = total_amounts;
    full.total_mse_conv = total_mse_conv;
    current_dir = "C:\Users\DrorSchein\Desktop\thesis\thesis\py to matlab";
    base_path = append(current_dir, "\success rate 3 repeats 10,000 each gs,net single,net multi ");
    if strcmp(mode,'integers')
        base_path = append(base_path, 'integers ');
    end
    base_path = append(base_path, int2str(depth), int2str(cycles), ' cycles.mat');
    save(base_path,'full')
   
 end
