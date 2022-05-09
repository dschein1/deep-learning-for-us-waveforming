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
    min_distance = 5 *  Diffraction_limit/pitch; % in units of the vector
    lower = round(N/2 - 16e-3/pitch);
    upper = round(N/2 + 16e-3/pitch) - 1;
    possible_indexes_base = 1:(upper - lower);
    for i = 1:batch_size
        points = zeros(1,number_in_each(i)) - 100;
        line = zeros(1,N);
        j=1;
        while j<=number_in_each(i)
             
            point = randi([lower,upper]);
            if all(abs(points - point)>min_distance)
                points(1,j) = point;
                j = j + 1;
            end
        end
        line(points) = 1;
        patterns(i,:) = line;
    end
end