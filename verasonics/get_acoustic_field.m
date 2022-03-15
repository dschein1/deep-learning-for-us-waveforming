function [info, data] = get_acoustic_field(L1,L3,d1,d3)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Script for sending commands to the Newport ESP300 motion controller and
    % read data from the Tektronix DPO4034
    % Intended use: 2D and 3D mapping of ultrasound pressure fields in
    % cartesian coordinates
    % Author: T. Grutman & R. Besso
    % Updated 07/09/2020
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %% Parameters for the scan %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global ESP301
    global MDO3024;

    tstart = tic;
    tic

    ESP301 = serial("COM5");
    set(ESP301, 'baudrate', 921600);
    set(ESP301, 'terminator', 13);


    MDO3024 = visa('agilent', 'USB0::0x0699::0x0408::C057185::0::INSTR');
    set(MDO3024, 'InputBufferSize', 100000);

    %%
%     L1 = 5; % length to cover along axis 1 (in mm)
%     L2 = 8; % length to cover along axis 2 (in mm)  
%     L3 = 0; % length to cover along axis 3 (in mm)
% 
%     d1 = 0.1; % resolution along axis 1 (in mm)
%     d2 = 0.1; % resolution along axis 2 (in mm)  
%     d3 = 1; % resolution along axis 3 (in mm)  
    L2 = 0;
    d2 = 1;
    Nx = round(L3/d3+1); % number of points for axis 3, Nx needs to be odd at the moment
    Ny = L1/d1+1; % number of points for axis 1
    Nz = L2/d2+1; % number of points for axis 2
    N = Nx*Ny*Nz; % total number of points

    data = [];

    % Set velocity, acceleration and deceleration for electric motors
    V = 4 ; % velocity (in mm/s)
    Ac = 20 ; % acceleration (in mm/s^2)
    Dc = 20 ; % deceleration (in mm/s^2)

    info = struct('GenInfo', '', 'xunit', struct('val', 0, 'info', ''),...
                  'xzero', struct('val', 0, 'info', ''),...
                  'ymult', struct('val', 0, 'info', ''),...
                  'yoff', struct('val', 0, 'info', ''),...
                  'yunit', struct('val', 0, 'info', ''),...
                  'yzero', struct('val', 0, 'info', ''));
    %%
    try
        fopen(MDO3024);
        fprintf(MDO3024, 'DATa:SOUrce CH1');
        fprintf(MDO3024, 'DATa:ENCdg SRIbinary');
        fprintf(MDO3024, 'WFMOutpre:BYT_Nr 2');
        fprintf(MDO3024, 'DATa:STARt 1'); %3001
        fprintf(MDO3024, 'DATa:STOP 10000'); %7999
        fprintf(MDO3024, 'WFMOutpre:WFId?');
        info.GenInfo = fscanf(MDO3024);

        fprintf(MDO3024, 'WFMOutpre:XUNit?');  
        info.xunit.val = fscanf(MDO3024); 
        info.xunit.info = 'Horizontal units';

        fprintf(MDO3024, 'WFMOutpre:XZEro?');
        info.xzero.val = fscanf(MDO3024);
        info.xzero.info = 'Time of the first point';

        fprintf(MDO3024, 'WFMOutpre:YMUlt?'); 
        info.ymult.val = fscanf(MDO3024);
        info.ymult.info = 'Vertical scale factor per digitizing level';

        fprintf(MDO3024, 'WFMOutpre:YOFf?');
        info.yoff.val = fscanf(MDO3024);
        info.yoff.info = 'vertical position in digitizing levels';

        fprintf(MDO3024, 'WFMOutpre:YUNit?');
        info.yunit.val = fscanf(MDO3024);
        info.yunit.info = 'Vertical units';

        fprintf(MDO3024, 'WFMOutpre:YZEro?'); 
        info.yzero.val = fscanf(MDO3024);
        info.yzero.info = 'Vertical offset';

        fclose(MDO3024);

        %%

        fopen(ESP301);

        %%

        fprintf(ESP301,'1MO'); % set Motor ON on axis 1
        fprintf(ESP301,'2MO'); % set Motor ON on axis 2
        fprintf(ESP301,'3MO'); % set Motor ON on axis 3
        fprintf(ESP301,['1VA',num2str(V)]); % set velocity on axis 1
        fprintf(ESP301,['2VA',num2str(V)]); % set velocity on axis 2
        fprintf(ESP301,['3VA',num2str(V)]); % set velocity on axis 3
        fprintf(ESP301,['1AC',num2str(Ac)]); % set acceleration on axis 1
        fprintf(ESP301,['2AC',num2str(Ac)]); % set acceleration on axis 2
        fprintf(ESP301,['3AC',num2str(Ac)]); % set acceleration on axis 3
        fprintf(ESP301,['1AG',num2str(Dc)]); % set deceleration on axis 1
        fprintf(ESP301,['2AG',num2str(Dc)]); % set deceleration on axis 2
        fprintf(ESP301,['3AG',num2str(Dc)]); % set deceleration on axis 3

        %%

        count = 0;
        h = waitbar(0,'Acquisition in progress...');
        s = clock;
        for nz = 1:Nz % axis z = axis 2
            posz = -(nz-1)*d2;
            moveESP300(ESP301,2,posz+L2/2);

            for nx = 1:Nx % axis x = axis 3

                if round(nz/2)~=nz/2 % odd indices, move forward
                    posx = (nx-1)*d3;
                    indx = nx;
                else % even indices, move backward
                    posx = (Nx-nx)*d3;
                    indx = Nx-nx+1;
                end
                moveESP300(ESP301,3,posx-L3/2);
                fopen(MDO3024);
                for ny = 1:Ny % axis y = axis 1

                    if round(nx/2)~=nx/2% odd indices, move forward
                        posy = (ny-1)*d1;
                        indy = ny;
                    else % even indices, move backward
                        posy = (Ny-ny)*d1;
                        indy = Ny-ny+1;
                    end

                    moveESP300(ESP301,1,posy-L1/2);

                    % read data on channel1 on DPO4034

                    fprintf(MDO3024, 'Curve?');
                    data_buf = binblockread(MDO3024, 'int16'); 

                    data(indx,indy,nz).signal = data_buf;%ones(1,10000)

%                     if ny == 1
%                         is = etime(clock,s);
%                         esttime = is * Ny * Nz;
%                     end
%                     h = waitbar(count/N,h,...
%                         ['Remaining time = ',num2str(esttime-etime(clock,s)/60,'%4.1f'),' min' ]);
                        
                    count = count+1;
                    waitbar(count/N,h);

                    vectx(count)=posx;
                    vecty(count)=posy;
                    vectz(count)=posz;
                end
                clrdevice(MDO3024);
                fclose(MDO3024);
            end
            if nz ==1
                is = etime(clock,s);
                esttime = is * Nz;
            end
            h = waitbar(nz/Nz,h,...
                ['Remaining time = ',num2str((esttime-etime(clock,s))/60,'%4.1f'),' min' ]);
        end
        close(h);

        %%

        moveESP300(ESP301,1,0); % move to origin
        moveESP300(ESP301,2,0); % move to origin
        moveESP300(ESP301,3,0); % move to origin
        fprintf(ESP301,'1MF'); % set Motor OFF on axis 1
        fprintf(ESP301,'2MF'); % set Motor OFF on axis 2
        fprintf(ESP301,'3MF'); % set Motor OFF on axis 3

        %%

        fclose(ESP301);
    catch
        fclose(ESP301);
        clrdevice(MDO3024);
        fclose(MDO3024);
        close(h)
    end
     function moveESP300(ESP300,axis,abs_pos)

    fprintf(ESP300,[num2str(axis),'PA',num2str(abs_pos)]); % move to absolute position (in mm)

    fprintf(ESP300,'TS');
    status  = fscanf(ESP300,'%s');
    while ~(strcmp(status,'P')) % ask status until position is reached
        fprintf(ESP300,'TS');
        status  = fscanf(ESP300,'%s');  % read status (Q=in motion, P=stopped)
        pause(0.01)
    end
    if status == 'P'
        fprintf(ESP300,'1TP'); % ask absolute position
        pos = fscanf(ESP300,'%s'); % read absolute position (in mm)
    end

     end
end


%%%%%%%%%%%%%%%%% instrreset %%%%%%%%%%%%%%%%%%%%%%%
