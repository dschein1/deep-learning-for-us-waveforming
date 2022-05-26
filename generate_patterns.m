function patterns = generate_patterns(batch_size,N, num_focus)
    patterns = zeros(batch_size,N);
    number_in_each = randi(2 ^ (num_focus - 1),batch_size,1);
    number_in_each = floor(log2(number_in_each)) + 1;
    %number_in_each = ones(batch_size,1);
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3; % 
    DX = (-30e-3+30e-3)/N;
    DZ = 40e-3; % Distance to pattern 
    Number_of_Elements = 128; % 
    Transducer_size = pitch*Number_of_Elements;
    Diffraction_limit=1.22*Wavelength*DZ/Transducer_size;
    min_distance = round(4.5 *  Diffraction_limit/pitch); % in units of the vector
    lower = round(N/2 - 16e-3/pitch);
    upper = round(N/2 + 16e-3/pitch) - 1;
    possible_indexes_base = lower:upper;
    parfor i = 1:batch_size
        points = zeros(1,number_in_each(i)) - 100;
        line = zeros(1,N);
        possible_indexes = possible_indexes_base;
        j=1;
        while j<=number_in_each(i)
            mask = possible_indexes ~= -1;
            actual_indexes = possible_indexes(mask);
            if length(actual_indexes) == 0
                break
            end
            point_index = randi(length(actual_indexes));
            %point = randi([lower,upper]);
            point = actual_indexes(point_index);
            index_in_real = find(possible_indexes == point,1);
            possible_indexes(max(index_in_real - min_distance,1):min(index_in_real + min_distance,length(possible_indexes))) = -1;
            points(1,j) = point;
%             if all(abs(points - point)>min_distance)
%                 points(1,j) = point;
%                 j = j + 1;
%             end
            j = j+1;
        end
        line(points) = 1;
        patterns(i,:) = line;
    end
end