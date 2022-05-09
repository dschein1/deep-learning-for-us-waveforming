function generate_data_from_field_ii()
    base_path = 'C:/Users/DrorSchein/Desktop/thesis/thesis/';
    init_field

    N = 1024;
    DZ = 40e-3; % Distance to pattern 
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3; % 
    path_to_py_dir = strcat(base_path,'py to matlab/');
    py_dir = dir(path_to_py_dir);
    last_step = -1;
    last_step_name = '';
    for i=1:size(py_dir,1)
        if contains(py_dir(i).name,'curriculum') & str2num(py_dir(i).name(11:12)) > last_step
            last_step = str2num(py_dir(i).name(11:12));
            last_step_name = py_dir(i).name;
        end
    end
    path_to_file = strcat(path_to_py_dir,last_step_name);
    data = load(path_to_file);
    data = data.from_net;
    num_samples = size(data,1);
    file_name = append('curriculum ',num2str(last_step));
    full_path = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/',file_name,'/base data/');
    %rmdir(full_path)
    mkdir(full_path)
    chunk_size = 2000;
    columns = cellstr(string(0:768));
    size(columns);
    num_iterations = ceil (num_samples / chunk_size);
    delays = zeros(chunk_size,128);
    amps = ones(chunk_size,128);
    results = zeros(chunk_size,512);
    for i=0:num_iterations - 1
        actual_size = min([chunk_size,num_samples - i *chunk_size]);
        index = double(i * chunk_size:i * chunk_size + actual_size - 1);
        delays(:,:) = double(data(index + 1,:));
        i
        %size(index)
        %size(amps)
        for j=1:actual_size
            delay = delays(j,:);
            delay = delay - min(min(delay));
            delay = unwrap(delay);
            delay = delay / (2 * pi);
            delay = delay / Frequancy;
            results(j,:) =  create_new_line(delay);
        
        %results(i,:) = calculateGS(patterns(i,:));
        %results(i,:) = abs(results(i,:));
        end
        
        full = [index' results amps delays];
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