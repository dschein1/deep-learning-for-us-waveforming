% function [out,pixel_out_size]=FSP_x(inp,Z,N,dx,lambda)
function [out]=FSP_X_near(inp,Z,N,dx,lambda)
% Free Space Propagation for all Distances (Z)

Z_c=N*dx^2/lambda; % Remains the same???
 
xx = -N/2:N/2-1;
 
% if Z<=Z_c
%  
    FFT_inp=fftshift(fft(fftshift(inp)));
    du=1/dx/N;
    PS=exp(1i*2*pi*Z/lambda*(sqrt(1-(xx*du*lambda).^2)));
    out=ifftshift(ifft(ifftshift(PS.*FFT_inp)));
    pixel_out_size=dx;
    
% else
% 
%     chirp_x=exp(pi*1i*dx^2*x.^2/lambda/Z);
%     df=lambda*abs(Z)/N/dx;
%     chirp_u=exp(pi*1i*df^2*x.^2/lambda/Z);
%     temp=inp.*chirp_x;
%     if Z>0
%         temp=fftshift(fft(fftshift(temp)))/N;
%     else
%         temp=fftshift(ifft(fftshift(temp)))*N;
%     end
%  
%     out=temp.*chirp_u;
%     pixel_out_size=df;
%     
% end
