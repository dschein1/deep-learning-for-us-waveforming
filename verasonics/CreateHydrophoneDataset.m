
delays_for_test = load('py to matlab\dataset.mat');
result = zeros(len(delays_for_test.from_net),512);
for i=1:len(delays_for_test.from_net)
    delay = delays_for_test.from_net(i,:);
    f = parfeval(run_flash_from_delay,0,delay);
    [a, tmp] = get_acoustic_field(35,1,0.218,1);
    result(i,:) = padarray(squeeze(tmp), (512 - round((35/0.218) + 1)) / 2,0); % puts zeros on both sides of the array, to speed up acquisition time
    cancel(f)
end

file_name = append(num2str(num_focus),' focus data delays real');

full_path = strcat('C:/Users/DrorSchein/Desktop/thesis/thesis/datasets/',file_name,'/base data/','0.parquet');
columns = cellstr(string(0:768));
index = 0:len(delays_for_test.from_net);
full = [index result ones(len(delays_for_test.from_net),128) delays_for_test.from_net];
full = array2table(full);
full.Properties.VariableNames = columns;
parquetwrite(full_path,full)

