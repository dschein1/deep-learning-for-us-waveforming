function res = extract_line(depth,source_im,dz)
    target = depth/dz;
    res = source_im(target,:);
    
end