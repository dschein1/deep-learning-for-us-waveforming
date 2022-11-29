function [lines_at_depth_net,lines_at_depth_gs]=compute_lines(pattern, delays_model, num_cycles, depth)
    patterns = padarray(pattern,[0 256],0,'both');
    [amps, delays_gs] = calculateGS(patterns,false);
    if num_cycles
        cycles = 1:10;
    else
        cycles = [1];
    end
    if depth
        depths = (30:50) .* 1e-3;
    else
        depths = [40e-3];
    end
    delays_gs = normalize_delays(delays_gs);
    delays_0 = normalize_delays(delays_model);
    lines_at_depth_gs = zeros(length(depths) * length(cycles),512);
    lines_at_depth_net = zeros(length(depths));
    for i=length(depths)
        lines_at_depth_net(i,:) = create_new_line(delays_0,ones(1,128),1,3,depths(i));
        for j=length(cycles)
            lines_at_depth_gs(i,:) = create_new_line(delays_gs,ones(1,128),cycles(i),depth(i));
        end
    end
    lines_at_depth_net = squeeze(lines_at_depth_net);
    lines_at_depth_gs = squeeze(lines_at_depth_gs);
end