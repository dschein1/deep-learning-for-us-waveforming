function Delay = calc_delay_dror(num_elements,c,focus,pitch)
%     num_points = size(focus,1);
%     base = -num_elements/2+1:num_elements/2;
%     for i=1:num_points
%         centers = base(round(num_elements/num_points)*(i-1) + 1:min(round(num_elements/num_points)*i,num_elements));
%         centers = centers * pitch;
%         centers = [centers' zeros(length(centers),2)];
%         center = centers(round(length(centers)/2),:);
%         shifted_point = focus(i,:) - center;
%         first = vecnorm(center - shifted_point);
%         second = vecnorm(centers - shifted_point,2,2);
%         Delay(round((num_elements/num_points))*(i-1) + 1:min(round((num_elements/num_points))*i,num_elements)) = (first - second)/c;
%     end
    
    
    first = norm(focus);
    centers = -num_elements/2+1:num_elements/2;
    centers = centers * pitch;
    centers = [centers' zeros(length(centers),2)];
    second = vecnorm(centers - focus,2,2);
    Delay = (first - second)/c;
end