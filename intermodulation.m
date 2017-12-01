%
% Non-linear Amplifier with 3rd order Intermodulation
%
clear all 
close all
clc


% Amplifier Characteristics
G = @(v_in) 5.62*v_in  - 4119 * v_in.^3;
G_lin = @(v_in) 5.62*v_in;
G_3rd = @(v_in) 4119 * v_in.^3;

% Get Signal Power
p_in = linspace(0,0.04,10000);
p_out =G(p_in);
p_out_lin = G_lin(p_in);
p_out_3rd = G_3rd(p_in);


% 1dB Compression point
C1db = find(10*log10(p_out_lin) - 10*log10(p_out) > 1,1);
C1db_in = 10*log10(p_in(C1db))+30
C1db_out = 10*log10(p_out(C1db)) + 30

% 3rd Order Interception Point
[~,IP3]=min(abs(p_out_lin(2:end) - p_out_3rd(2:end)));
IP3_in =10*log10(p_in(IP3))+30
IP3_out =10*log10(p_out_lin(IP3))+30

plot(10*log10(p_in)+30,10*log10(p_out)+30,...
    10*log10(p_in)+30,10*log10(p_out_lin)+30,...
    10*log10(p_in)+30,10*log10(p_out_3rd)+30,...
    C1db_in,C1db_out,'ko',...
    IP3_in,IP3_out,'kx')
ylim([-20 30])
xlabel('Input Power')
ylabel('Output Power')
grid on

