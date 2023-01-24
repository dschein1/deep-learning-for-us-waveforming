function generate_data(num_samples, allow_amps, depth, distribution)
    batch_size = num_samples;
    init_field
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3; % 
    num_focus = 10;
    file_name = append(num2str(num_focus),' focus data delays', distribution);
    if allow_amps == true
        file_name = append(file_name,' amps');
    end
    full_path = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/',file_name,'/base data/');
    if strcmp(distribution, 'uniform')
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/','curriculum 1 ', distribution, '/base data/');
    else    
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/','curriculum 1','/base data/');
    end
    if depth ~= 40e-3
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/dataset depth ',int2str(depth),'/base data/');
    end
    %rmdir(full_path)
    mkdir(full_path)    
    mkdir(full_path_curr)
    group_size = 1e5;
    number_of_groups = ceil(num_samples / group_size);
    for i=0:number_of_groups - 1
        actual_group_size = min([group_size, num_samples - i * group_size]);
        generate_group_of_samples(actual_group_size, i, allow_amps, depth, distribution);
        clear mex;
    end

end

function generate_group_of_samples(num_samples, group_index, allow_amps, depth, distribution)
    chunk_size = 1000;
    DZ = depth;
    vec_size = 1024;
    num_focus = 10;
    columns = cellstr(string(0:768));
    file_name = append(num2str(num_focus),' focus data delays', distribution);
    if allow_amps == true
        file_name = append(file_name,' amps');
    end
    full_path = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/',file_name,'/base data/');
    if strcmp(distribution, 'uniform')
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/','curriculum 1 ', distribution, '/base data/');
    else    
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/','curriculum 1','/base data/');
    end    
    if depth ~= 40e-3
        full_path_curr = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/dataset depth ',int2str(depth),'/base data/');
    end    
    [chunk_size num_samples num_samples/chunk_size group_index group_index * num_samples]
    num_iterations = ceil (num_samples / chunk_size);
    base_index = group_index * num_samples;
    parfor i=0:num_iterations - 1
        init_field
        actual_size = min([chunk_size,num_samples - i *chunk_size])
        index = base_index + (i * chunk_size:i * chunk_size + actual_size - 1);
        amps = zeros(actual_size,128);
        delays = zeros(actual_size,128);
        patterns = generate_patterns(actual_size,vec_size, num_focus, distribution);
        results = zeros(actual_size,512);
        for j=1:actual_size
            [amps(j,:) , delays(j,:)] = calculateGS(patterns(j,:),allow_amps, DZ);
            delays_normalized = normalize_delays(delays(j,:));
            results(j,:) = create_new_line(delays_normalized, ones(1,128), 1, 3, DZ);
        end
        full = [index' results amps delays];
        full = array2table(full);
        full.Properties.VariableNames = columns;
        chunk_index = i + group_index * num_iterations;
        name_to_write = strcat(full_path_curr,string(chunk_index),".parquet");
        parquetwrite(name_to_write,full)
        field_end();
        clear_mex();
    end
end

function clear_mex
    clear mex;
end