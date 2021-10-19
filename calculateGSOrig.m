function result = calculateGSOrig(desired_Output_Shift)
    N = 1024;
    Transducer = zeros(1,N);
    GS = 15;
    DZ = 40e-3; % Distance to pattern 
    Frequancy = 4.464e6; 
    v = 1490; % water in room temperature m/sec (in body  v = 1540)
    Wavelength = v/Frequancy;
    pitch = 0.218e-3; % 
    Number_of_Elements = 128; % 
    Output = desired_Output_Shift;
    for n = 1:GS
        Output = (desired_Output_Shift.^0.5).*exp(1i*angle(Output));
%         Output=circshift(Output,[0,3]);
        Transducer = FSP_X_near(Output,-DZ,N,pitch,Wavelength);
    %     Transducer = ones(size(Transducer)).*exp(1i*angle(Transducer)); % if you don't allow apodiztion
%         Transducer(1:ceil(N/2)-Number_of_Elements/2) = 0;     % set to zero the pixels outside the transducer
%         Transducer(floor(N/2)+Number_of_Elements/2:end) = 0; % set to zero the pixels outside the transducer
        Transducer(1:512-63) = 0;     % set to zero the pixels outside the transducer
        Transducer(512+64:end) = 0; % set to zero the pixels outside the transducer
        Output = FSP_X_near(Transducer,+DZ,N,pitch,Wavelength);
    end
    result = Output;
end