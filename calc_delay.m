function delay_vector = calc_delay(num_element,size_element,c,focus_point)

%  Procedure for setting the delay of individual
%  mathematical elements making up the transducer 
%
%  Parameters:  num_element    - Number of elements the transducer contains  
%               size_element   - Size of each element [m]
%               c              - Sound velocity in [m/sec]
%               focus_point    - [xf,yf,zf] coordinates of the focus point
%                                desired in [m]
% 
%  Return:      delay vector   - Vector of the delay that should be applied
%                                to each element [sec]
%
%  Version 1.0, September 29, 2020 by Raphael Abiteboul

xf=focus_point(1); % x coordinate of focus point
zf=focus_point(3); % z coordinate of focus point

% find the distance of each element from the center x=0
if mod(num_element,2)==1 % if the transducer has an odd number of elements
    x_element = [-floor(num_element/2):0 1:floor(num_element/2)]*size_element; 
else % if the transducer has an even number of elements
    x_element = [(-num_element/2+0.5):0 0.5:(num_element/2-0.5)]*size_element;
end

zmax = ((x_element(1)-(abs(xf)))^2 + zf^2)^0.5; % the distance of the farthest element to the focal point
dist_to_focus = ((zf)^2 + (x_element-xf).^2 ).^0.5 ; % distance of each element from the focus point
delay_vector = (zmax - dist_to_focus)./(c); % time[sec] = distance[m]/velocity[m/sec]

end