function patterns = generate_patterns(batch_size,N)
    patterns = zeros(batch_size,N);
    number_in_each = randi(1,batch_size,1);
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
    min_distance = 3 *  Diffraction_limit/pitch; % in units of the vector
    lower = round(N/2 - 30e-3/pitch);
    upper = round(N/2 + 30e-3/pitch);
    for i = 1:batch_size
        points = zeros(1,number_in_each(i)) - 100;
        j=1;
        while j<=number_in_each(i)
            point = randi([lower,upper]);
            if all(abs(points - point)>min_distance)
                points(1,j) = point;
                j = j + 1;
            end
        end
        point
        patterns(i,points) = 1;
    end
end