function generate_data(num_samples,vec_size,num_focus,allows_amps)
    batch_size = num_samples;
    
    N = vec_size;
    DZ = 40e-3; % Distance to pattern 
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3; % 
    file_name = append(num2str(num_focus),' focus data delays');
    if allows_amps == true
        file_name = append(file_name,' amps');
    end
    full_path = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/',file_name,'/base data/');
    %rmdir(full_path)
    mkdir(full_path)
    chunk_size = 125000
    columns = cellstr(string(0:768));
    size(columns);
    num_iterations = ceil (num_samples / chunk_size);
    
    parfor i=0:num_iterations - 1
        actual_size = min([chunk_size,num_samples - i *chunk_size]);
        index = i * chunk_size:i * chunk_size + actual_size - 1;
        %size(index)
        amps = zeros(actual_size,128);
        %size(amps)
        delays = zeros(actual_size,128);
        patterns = generate_patterns(actual_size,vec_size, num_focus);
        %results = zeros(size,256);
        for j=1:actual_size
            [amps(j,:) , delays(j,:)] = calculateGS(patterns(j,:),allows_amps);
        %results(i,:) = calculateGS(patterns(i,:));
        %results(i,:) = abs(results(i,:));
        end
        full = [index' patterns(:,512-512/2 + 1:512+512/2) amps delays];
        full = array2table(full);
        full.Properties.VariableNames = columns;
        i
        size(full)
        name_to_write = strcat(full_path,string(i),".parquet");
        %size(full)
        %parquetwrite(full_path,full,'')
        parquetwrite(name_to_write,full)
        %writetable(full,name_to_write)
        %table_to_return = full;
    end
end