function result = generate_pressure_line(Th,depth)
    x_min = -15e-3;
    x_max = 15e-3;
    x = linspace(x_min,x_max,200);
    point = [0 0 depth];
    p = zeros(1,200);
    for ii=1:length(x)
        clc; ii
        point(1)=x(ii);
        [temp,~] = calc_hp(Th,point);
        p(ii) = norm(temp);
    end
    result = abs(p);
    result = result - min(min(result));
    result = result/max(max(result));    
end