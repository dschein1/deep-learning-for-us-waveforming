% Notice: 
%   This file is provided by Verasonics to end users as a programming
%   example for the Verasonics Vantage Research Ultrasound System.
%   Verasonics makes no claims as to the functionality or intended
%   application of this program and the user assumes all responsibility 
%   for its use
%
% File name: SetUpP6_3_128RyLns.m - Example of phased array imaging with 
%                                    focus transmits
% Description: 
%   Sequence programming file for P6-3 phased array in virtual apex format, 
%   using 128 ray lines(focus transmits) and receive acquisitions. All 128 
%   transmit and receive channels are active for each acquisition. The 
%   receive acquisitions use 100% bandwidth to improve DMA transfers. 
%   Processing is asynchronous with respect to acquisition. Note: The P6-3 
%   is a 128 element probe that is wired to the scanhead connector with
%   element 0-31 connected to inputs 1-32, and elements 32-63 connected to
%   input 97-128. We therefore need a Trans.Connector array to 
%   specify the connector channels used, which will be defined by the 
%   computeTrans function.  
%
% Last update:
% 12/13/2015 - modified for SW 3.0

clear all

% Specify system parameters.
Resource.Parameters.numTransmit = 128;  % number of transmit channels.
Resource.Parameters.numRcvChannels = 128;  % number of receive channels.
Resource.Parameters.speedOfSound = 1490;
Resource.Parameters.speedCorrectionFactor = 1.0;
Resource.Parameters.verbose = 1;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 0;
%  Resource.Parameters.simulateMode = 1 forces simulate mode, even if hardware is present.
%  Resource.Parameters.simulateMode = 2 stops sequence and processes RcvData continuously.

P.numRays = 256;   % 128 no. of Rays (1 for Flash transmit)
startDepth = 0; %[mm]
endDepth = 80; %[mm]
Trans.frequency = 4.4643; %[MHz]
P.startDepth = (startDepth*1e-3)/Resource.Parameters.speedOfSound*Trans.frequency*1e6;
P.endDepth = (endDepth*1e-3)/Resource.Parameters.speedOfSound*Trans.frequency*1e6;
P.txFocus = (70*1e-3)/Resource.Parameters.speedOfSound*Trans.frequency*1e6; % initial value of P.txFocus

% Specify Trans structure array.
Trans.name = 'P6-3';
Trans.units = 'wavelengths'; % required in Gen3 to prevent default to mm units
Trans = computeTrans(Trans);
Trans.maxHighVoltage = 50;  % set maximum high voltage limit for pulser supply.

P.theta = -pi/4;
P.rayDelta = 2*(-P.theta)/(P.numRays-1);
P.aperture = Trans.numelements*Trans.spacing;
P.radius = 0;%(P.aperture/2)/tan(-P.theta); % dist. to virt. apex

% Specify PData structure array.
PData(1).PDelta = [0.3, 0, 0.3];  % x, y, z pdeltas
PData(1).Size(1) = 10 + ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));
PData(1).Size(2) = 10 + ceil(2*(P.endDepth + P.radius)*sin(-P.theta)/PData(1).PDelta(1));
PData(1).Size(3) = 1;      % single image page
PData(1).Origin = [-(PData.Size(2)/2)*PData(1).PDelta(1),0,P.startDepth];
% - specify 128 Region structures.
PData(1).Region = repmat(struct('Shape',struct( ...
                    'Name','SectorFT',...
                    'Position',[0,0,-P.radius],...
                    'z',P.startDepth,...
                    'r',P.radius+P.endDepth,...
                    'angle',P.rayDelta,...
                    'steer',0)),1,P.numRays);
% - set position of regions to correspond to beam spacing.
for i = 1:P.numRays
    PData(1).Region(i).Shape.steer(1) = P.theta + (i-1)*P.rayDelta;
end
PData(1).Region = computeRegions(PData(1));

% Specify Resources.
Resource.RcvBuffer(1).datatype = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = 2048*P.numRays; % This is for the max range on range slider. 
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames = 8;  
Resource.InterBuffer(1).numFrames = 1;  % 1 frame defined but no intermediate buffer needed.
Resource.ImageBuffer(1).numFrames = 10;
Resource.DisplayWindow(1).Title = 'P6-3_128RyLns';
Resource.DisplayWindow(1).pdelta = 0.4;
ScrnSize = get(0,'ScreenSize');
DwWidth = ceil(PData(1).Size(2)*PData(1).PDelta(1)/Resource.DisplayWindow(1).pdelta);
DwHeight = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);
Resource.DisplayWindow(1).Position = [250,(ScrnSize(4)-(DwHeight+150))/2, ...  % lower left corner position
                                      DwWidth, DwHeight];
Resource.DisplayWindow(1).ReferencePt = [PData(1).Origin(1),0,PData(1).Origin(3)];   % 2D imaging is in the X,Z plane
Resource.DisplayWindow(1).numFrames = 20;
Resource.DisplayWindow(1).AxesUnits = 'mm';
Resource.DisplayWindow.Colormap = gray(256);

% Specify TX structure array.
TX = repmat(struct('waveform', 1, ...
                   'Origin', [0.0,0.0,0.0], ...
                   'focus', P.txFocus, ...
                   'Steer', [0.0,0.0], ...
                   'Apod', ones(1,Trans.numelements), ...  % set TX.Apod for 128 elements
                   'Delay', zeros(1,Trans.numelements)), 1, P.numRays);
% - Set event specific TX attributes.
Angles = P.theta:P.rayDelta:(P.theta + (P.numRays-1)*P.rayDelta);
TXorgs = P.radius*tan(Angles);

% Specify TW structure array.
TW(1).type = 'parametric';
TW(1).Parameters = [Trans.frequency,.67,2,1];   % A, B, C, D

Pitch = 0.218e-3; % Element size [m]
Fill_Factor = 1;
width = Pitch*Fill_Factor;
addpath('C:\Users\Administrator\Documents\MATLAB\Raphael');
vector_delay = calc_delay(128,width,1490,[0,0,70]/1000)*Trans.frequency*1e6; % wavelengths units
% vector_delay = calc_delay(128,width,1540,[5,0,50]/1000)*1540e3; % mm units

for n = 1:P.numRays   % P.numRays transmit events
    TX(n).Origin = [TXorgs(n),0.0,0.0];
    TX(n).Steer = [Angles(n),0.0];
    TX(n).Delay = computeTXDelays(TX(n));
end

% Specify Receive structure arrays. 
% - We need P.numRays Receives for each frame.
% -- Compute the maximum receive path length, using the law of cosines.
maxAcqLength = ceil(sqrt(P.aperture^2 + P.endDepth^2 - 2*P.aperture*P.endDepth*cos(P.theta-pi/2)) - P.startDepth);
wlsPer128 = P.numRays/(2*2); % wavelengths in 128 samples for 2 samplesPerWave
Receive = repmat(struct('Apod', ones(1,Trans.numelements), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', P.startDepth + wlsPer128*ceil(maxAcqLength/wlsPer128), ...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode', 'BS100BW', ...
                        'mode', 0, ...
                        'callMediaFunc', 0),1,P.numRays*Resource.RcvBuffer(1).numFrames);
% - Set event specific Receive attributes.
for i = 1:Resource.RcvBuffer(1).numFrames
    Receive(P.numRays*(i-1)+1).callMediaFunc = 1;
    for j = 1:P.numRays
        Receive(P.numRays*(i-1)+j).framenum = i;
        Receive(P.numRays*(i-1)+j).acqNum = j; 
    end
end

% Specify TGC Waveform structure.
% TGC.CntrlPts = [0,440,598,658,666,727,736,847];
TGC.CntrlPts = [0,143,286,429,571,714,857,1000]; % linear TGC
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Recon structure array.
Recon = struct('senscutoff', 0.5, ...
               'pdatanum', 1, ...
               'rcvBufFrame',-1, ...
               'IntBufDest', [1,1], ...
               'ImgBufDest', [1,-1], ...
               'RINums', 1:P.numRays);

% Define ReconInfo structures.
ReconInfo = repmat(struct('mode', 'replaceIntensity', ...  % replace data.
                   'txnum', 1, ...
                   'rcvnum', 1, ...
                   'regionnum', 0), 1, P.numRays);
% - Set specific ReconInfo attributes.
for i = 1:P.numRays
    ReconInfo(i).txnum = i;
    ReconInfo(i).rcvnum = i;
    ReconInfo(i).regionnum = i;
end
    
% Specify Process structure array.
pers = 20;
Process(1).classname = 'Image';
Process(1).method = 'imageDisplay';
Process(1).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'pgain',1.0,...            % pgain is image processing gain
                         'reject',2,...      % reject level 
                         'persistMethod','simple',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...  %method of interp. (1=4pt)
                         'grainRemoval','none',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','log',...
                         'compressFactor',60,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',1};
                     
% Specify SeqControl structure arrays.
t1 = 2*354*.4 + 20; % ray line acquisition time for worst case range in usec
t2 = round((1e+06-127*t1*25)/25);   % Time between frames at 25 fps.
SeqControl(1).command = 'jump'; %  - Jump back to start.
SeqControl(1).argument = 1;
SeqControl(2).command = 'timeToNextAcq';  % set time between rays
SeqControl(2).argument = t1; 
SeqControl(3).command = 'timeToNextAcq';  % set time between frames
SeqControl(3).argument = t2;
SeqControl(4).command = 'triggerOut';
SeqControl(5).command = 'returnToMatlab';

nsc = length(SeqControl)+1; % nsc is count of SeqControl objects

% Specify Event structure arrays.
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    for j = 1:P.numRays                      % Acquire rays
        Event(n).info = 'Acquire ray line';
        Event(n).tx = j; 
        Event(n).rcv = P.numRays*(i-1)+j;   
        Event(n).recon = 0;      % no reconstruction.
        Event(n).process = 0;    % no processing
        Event(n).seqControl = 2; % time between rays
        n = n+1;
    end
    Event(n-1).seqControl = [3,nsc]; % Replace last event's seqControl for frame time and transferToHost.
       SeqControl(nsc).command = 'transferToHost'; % transfer frame to host buffer
       nsc = nsc+1;

    Event(n).info = 'recon and process'; 
    Event(n).tx = 0;         % no transmit
    Event(n).rcv = 0;        % no rcv
    Event(n).recon = 1;      % reconstruction
    Event(n).process = 1;    % process
    if floor(i/3) == i/3     % Exit to Matlab every 3rd frame reconstructed 
        Event(n).seqControl = 4;
    else
        Event(n).seqControl = 0;
    end
    n = n+1;
end

Event(n).info = 'Jump back';
Event(n).tx = 0;        % no TX
Event(n).rcv = 0;       % no Rcv
Event(n).recon = 0;     % no Recon
Event(n).process = 0; 
Event(n).seqControl = 1;


% User specified UI Control Elements
% - Sensitivity Cutoff
UI(1).Control =  {'UserB7','Style','VsSlider','Label','Sens. Cutoff',...
                  'SliderMinMaxVal',[0,1.0,Recon(1).senscutoff],...
                  'SliderStep',[0.025,0.1],'ValueFormat','%1.3f'};
UI(1).Callback = text2cell('%SensCutoffCallback');

% - Range Change
wls2mm = 1;
AxesUnit = 'wls';
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        AxesUnit = 'mm';
        wls2mm = Resource.Parameters.speedOfSound/1000/Trans.frequency;
    end
end
UI(2).Control = {'UserA1','Style','VsSlider','Label',['Range (',AxesUnit,')'],...
                 'SliderMinMaxVal',[64,300,P.endDepth]*wls2mm,'SliderStep',[0.1,0.2],'ValueFormat','%3.0f'};
UI(2).Callback = text2cell('%RangeChangeCallback');             
             
% - Transmit focus change
UI(3).Control = {'UserB4','Style','VsSlider','Label',['TX Focus (',AxesUnit,')'],...
                 'SliderMinMaxVal',[20,320,P.txFocus]*wls2mm,'SliderStep',[0.1,0.2],'ValueFormat','%3.0f'};
UI(3).Callback = text2cell('%TxFocusCallback');

% - Save IQ Pushbutton
UI(4).Control = {'UserC1','Style','VsPushButton','Label','Save IQ'};
UI(4).Callback = text2cell('%saveIQbutton');

% Specify factor for converting sequenceRate to frameRate.
frameRateFactor = 3;

% Save all the structures to a .mat file.
save('MatFiles/P6_3_Bmode');
filename = 'P6_3_Bmode';
VSX
return

% **** Callback routines to be converted by text2cell function. ****
%SensCutoffCallback - Sensitivity cutoff change
ReconL = evalin('base', 'Recon');
for i = 1:size(ReconL,2)
    ReconL(i).senscutoff = UIValue;
end
assignin('base','Recon',ReconL);
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'Recon'};
assignin('base','Control', Control);
return
%SensCutoffCallback

%RangeChangeCallback - Range change
simMode = evalin('base','Resource.Parameters.simulateMode');
% No range change if in simulate mode 2.
if simMode == 2
    set(hObject,'Value',evalin('base','P.endDepth'));
    return
end
Trans = evalin('base','Trans');
Resource = evalin('base','Resource');
scaleToWvl = Trans.frequency/(Resource.Parameters.speedOfSound/1000);

P = evalin('base','P');
P.endDepth = UIValue;
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        P.endDepth = UIValue*scaleToWvl;    
    end
end
assignin('base','P',P);

PData = evalin('base','PData');
PData(1).Size(1) = 10 + ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));
% - specify 128 Region structures.
PData(1).Region = repmat(struct('Shape',struct( ...
                    'Name','SectorFT',...
                    'Position',[0,0,-P.radius],...
                    'z',P.startDepth,...
                    'r',P.radius+P.endDepth,...
                    'angle',P.rayDelta,...
                    'steer',0)),1,P.numRays);
% - set position of regions to correspond to beam spacing.
for i = 1:P.numRays
    PData(1).Region(i).Shape.steer(1) = P.theta + (i-1)*PData(1).Region(i).Shape.angle;
end
PData(1).Region = computeRegions(PData(1));
assignin('base','PData',PData);
evalin('base','Resource.DisplayWindow(1).Position(4) = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);');
Receive = evalin('base', 'Receive');
maxAcqLength = ceil(sqrt(P.aperture^2 + P.endDepth^2 - 2*P.aperture*P.endDepth*cos(P.theta-pi/2)) - P.startDepth);
wlsPer128 = P.numRays/(2*2);
for i = 1:size(Receive,2)
    Receive(i).endDepth = P.startDepth + wlsPer128*ceil(maxAcqLength/wlsPer128);
end
assignin('base','Receive',Receive);
evalin('base','TGC.rangeMax = P.endDepth;');
evalin('base','TGC.Waveform = computeTGCWaveform(TGC);');
evalin('base','if VDAS==1, Result = loadTgcWaveform(1); end');
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'PData','InterBuffer','Receive','Recon','DisplayWindow','ImageBuffer'};
assignin('base','Control', Control);
assignin('base', 'action', 'displayChange');
return
%RangeChangeCallback

%TxFocusCallback - TX focus changel
simMode = evalin('base','Resource.Parameters.simulateMode');
% No focus change if in simulate mode 2.
if simMode == 2
    set(hObject,'Value',evalin('base','P.txFocus'));
    return
end
Trans = evalin('base','Trans');
Resource = evalin('base','Resource');
scaleToWvl = Trans.frequency/(Resource.Parameters.speedOfSound/1000);

P = evalin('base','P');
P.txFocus = UIValue;
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        P.txFocus = UIValue*scaleToWvl;    
    end
end
assignin('base','P',P);

% - Redefine event specific TX attributes for the new focus.
TX = evalin('base', 'TX');
for n = 1:P.numRays   % 128 transmit events
    % write new focus value to TX
    TX(n).focus = P.txFocus;
    TX(n).Delay = computeTXDelays(TX(n));
end
assignin('base','TX', TX);
% Set Control command to update TX
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'TX'};
assignin('base','Control', Control);
return
%TxFocusCallback

%saveIQbutton - save IQ
if evalin('base','freeze')==0   % no action if not in freeze
    msgbox('Please freeze VSX');
    return
end

Control.Command = 'copyBuffers';
runAcq(Control); % NOTE:  If runAcq() has an error, it reports it then exits MATLAB.

IQfilename = ['IQdata_',datestr(now,'dd-mmmm-yyyy_HH-MM-SS')];

RcvLastFrame = size(RcvData,3);
if (~evalin('base','simButton'))
    RcvLastFrame = Resource.RcvBuffer(1).lastFrame;
end

[fn,pn,~] = uiputfile('*.mat','Save IQ data as',IQfilename);
if ~isequal(fn,0) % fn will be zero if user hits cancel
    fn = strrep(fullfile(pn,fn), '''', '''''');
    save(fn,'IData','QData','RcvLastFrame','-v7.3');
    fprintf('The IQ data has been saved at %s \n',fn);
else
    disp('The IQ data is not saved.');
end

return
%saveIQbutton