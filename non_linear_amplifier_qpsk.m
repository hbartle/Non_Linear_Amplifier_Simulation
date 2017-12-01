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


%% Create the Time domain QPSK signal

% Bit Rate
R = 1e6;

% Create Data (300)
disp('Creating random data...')
number_of_bits=50;
data=randi(2,number_of_bits,1)-1;
data_NZR=2*data-1; 
disp('Done!')

% Encode using Non-Return-To-Zero Coding
data_iq=reshape(data_NZR,2,length(data)/2); 

% RRC Baseband Filter
b = rcosdesign(0.35,6,2);
x_i = upfirdn(data_iq(1,:), b, 2);
x_q = upfirdn(data_iq(2,:), b, 2);
x_qpsk = [x_i;x_q];

x_bpsk = upfirdn(data_NZR,b,2);


% QPSK 
figure('units','normalized','outerposition',[0 0 1 1])
% Plot Baseband Signal
t_bit = linspace(0,number_of_bits*1/R,number_of_bits);
subplot(2,2,1)
stairs(t_bit,data)
grid on;
title('Data');
ylim([0 1.5]);

subplot(2,2,3)
stairs(t_bit,data_NZR)
title('BPSK Baseband')
grid on
ylim([-1.5 1.5]);

subplot(2,2,2)
stairs(t_bit,data)
grid on;
title('Data');
ylim([0 1.5]);

t_bit = linspace(0,number_of_bits/2*1/R,length(x_bpsk));
subplot(2,2,4)
plot(t_bit,x_bpsk)
title('BPSK Baseband with RRC')
grid on
ylim([-1.5 1.5]);




%%%%%%%%%%%%%%%
% QPSK 
figure('units','normalized','outerposition',[0 0 1 1])
% Plot Baseband Signal
t_bit = linspace(0,number_of_bits*1/R,number_of_bits);
subplot(3,2,1)
stairs(t_bit,data)
grid on;
title('Data');
ylim([0 1.5]);

subplot(3,2,3)
stairs(t_bit(1:length(data_iq)),data_iq(1,:))
title('I Component')
grid on
ylim([-1.5 1.5]);

subplot(3,2,5)
stairs(t_bit(1:length(data_iq)),data_iq(2,:))
title('Q Component')
grid on
ylim([-1.5 1.5]);


% Plot shaped Baseband Signal
t_bit = linspace(0,number_of_bits*1/R,number_of_bits);
subplot(3,2,2)
stairs(t_bit,data)
grid on;
title('Data');
ylim([0 1.5]);

t_bit = linspace(0,number_of_bits/2*1/R,length(x_i));
subplot(3,2,4)
plot(t_bit,x_i)
title('I Component with baseband RRC')
grid on
ylim([-1.5 1.5]);

subplot(3,2,6)
plot(t_bit,x_q)
title('Q Component with baseband RRC')
grid on
ylim([-1.5 1.5]);










%% Carrier Properties
% Amplitude
A = linspace(0,0.038,200);

% Carrier Frequency
f0=10e6;
% Bit duration
T=1/R; 
% Sampling Factor
k_sampling = 100;
% Time vector for one bit information
t=T/k_sampling:T/k_sampling:T;
% Time vector for output signal
t_bpsk=T/k_sampling:T/k_sampling:(T*length(data_NZR));
t_bpsk_rrc=T/k_sampling:T/k_sampling:(T*length(x_bpsk));
t_qpsk=T/k_sampling:T/k_sampling:(T*length(data_iq));
t_qpsk_rrc=T/k_sampling:T/k_sampling:(T*length(x_qpsk));

% Signal Vector with different Amplitudes
v_bpsk = nan*ones(length(A),length(t_bpsk));
w_psk = nan*ones(length(A),length(t_bpsk_rrc));
v_qpsk = nan*ones(length(A),length(t_qpsk));
w_qpsk = nan*ones(length(A),length(t_qpsk_rrc));


disp('BPSK Modulation...')
% BPSK Modulation
for k=1:length(A)
    y=[];
    for i=1:length(data_NZR)
        y= [y A(k) * data_NZR(i)*sin(2*pi*f0*t)]; 
    end
    % Store Signal in big matrix
    v_bpsk(k,:)=  y;
end
disp('Done!')
disp('BPSK Modulation with RRC...')
% BPSK Modulation w/ RRC
for k=1:length(A)
    y=[];
    for i=1:length(x_bpsk)
        y= [y A(k) * x_bpsk(i)*sin(2*pi*f0*t)]; 
    end
    % Store Signal in big matrix
    w_bpsk(k,:)=  y;
end
disp('Done!')
disp('QPSK Modulation...')
% QPSK Modulation
for k=1:length(A)
    y_i=[];
    y_q=[];
    for i=1:length(data_iq)
        % inphase component
        y1=A(k) * data_iq(1,i)*cos(2*pi*f0*t); 
        % Quadrature component
        y2= A(k) * data_iq(2,i)*sin(2*pi*f0*t);
        % Inphase signal vector
        y_i = [y_i y1]; 
        % Quadrature signal vector
        y_q = [y_q y2]; 

    end
    % Modulated signal vector
    y = y_i + y_q;
    % Store Signal in big matrix
    v_qpsk(k,:)=  y;
end
disp('Done!')
disp('QPSK Modulation with RRC...')
% QPSK Modulation w/ RRC
for k=1:length(A)
    y_i=[];
    y_q=[];
    for i=1:length(x_qpsk)
        % inphase component
        y1=A(k) * x_qpsk(1,i)*cos(2*pi*f0*t); 
        % Quadrature component
        y2=A(k) * x_qpsk(2,i)*sin(2*pi*f0*t);
        % Inphase signal vector
        y_i = [y_i y1]; 
        % Quadrature signal vector
        y_q = [y_q y2]; 

    end
    % Modulated signal vector
    y = y_i + y_q;
    % Store Signal in big matrix
    w_qpsk(k,:)=  y;
end
disp('Done!')


figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,1,1)
stem(data(1:5),'MarkerSize',12,'LineWidth',2)
title('Data')
grid on 
ylim([0 1.2])
subplot(3,1,2)
plot(v_bpsk(end,1:4*length(t)));
title('BPSK Modulated Data')
%ylim([-1 1])
grid on 
subplot(3,1,3)
plot(v_qpsk(end,1:2*length(t)));
title('QPSK Modulated Data')
%ylim([-1 1])
grid on 



%% Amplifier Analysis
disp('Amplify Signals...')
% Amplifier Signal
v_bpsk_out = G(v_bpsk);
w_bpsk_out = G(w_bpsk);
v_bpsk_out_lin = G_lin(v_bpsk);
v_bpsk_out_3rd = G_3rd(v_bpsk);

v_qpsk_out = G(v_qpsk);
w_qpsk_out = G(w_qpsk);
v_qpsk_out_lin = G_lin(v_qpsk);
v_qpsk_out_3rd = G_3rd(v_qpsk);

% Calculate Powers in dB
p_bpsk_in = 10*log10(rms(v_bpsk,2).^2)+30;
p_bpsk_rrc_in = 10*log10(rms(w_bpsk,2).^2)+30;
p_bpsk_out = 10*log10(rms(v_bpsk_out,2).^2)+30;
p_bpsk_rrc_out = 10*log10(rms(w_bpsk_out,2).^2)+30;
p_bpsk_out_lin = 10*log10(rms(v_bpsk_out_lin,2).^2)+30;
p_bpsk_out_3rd = 10*log10(rms(v_bpsk_out_3rd,2).^2)+30;

p_qpsk_in = 10*log10(rms(v_qpsk,2).^2)+30;
p_qpsk_rrc_in = 10*log10(rms(w_qpsk,2).^2)+30;
p_qpsk_out = 10*log10(rms(v_qpsk_out,2).^2)+30;
p_qpsk_rrc_out = 10*log10(rms(w_qpsk_out,2).^2)+30;
p_qpsk_out_lin = 10*log10(rms(v_qpsk_out_lin,2).^2)+30;
p_qpsk_out_3rd = 10*log10(rms(v_qpsk_out_3rd,2).^2)+30;
disp('Done!')

%1dB Compression point
C1db = find(p_qpsk_out_lin - p_qpsk_out > 1,1);
C1db_in = p_qpsk_in(C1db);
C1db_out = p_qpsk_out(C1db);

% 3rd Order Interception Point
[~,IP3]=min(abs(p_qpsk_out_lin(2:end) - p_qpsk_out_3rd(2:end)));
IP3_in = p_qpsk_in(IP3);
IP3_out = p_qpsk_out_lin(IP3);

figure('units','normalized','outerposition',[0 0 1 1])
plot(p_qpsk_in,p_qpsk_out,...
     p_qpsk_in,p_qpsk_out_lin,'r',...
     p_qpsk_in,p_qpsk_out_3rd,'g')
ylim([-40 20])
xlim([-inf inf])
title('Amplifier Characteristic')
xlabel('Input Power [dBm]')
ylabel('Output Power [dBm]')
legend('Amplifier Output','Linear','3rd Intermodulation','Location','NW')
grid on

%% Spectral Analysis


% Parameters
BW = 1e5; 
f_DAC = 10*f0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BPSK
disp('Calculate BPSK spectrum...')
[~,number_of_samples]= size(v_bpsk);
window_size = number_of_samples;
f = ((0: window_size - 1)-window_size/2)*(f_DAC/window_size);

% Input Signal
v_fft_in = abs(fftshift(fft(v_bpsk(2,:),window_size))).^2;
npsd_in = v_fft_in/max(v_fft_in);

% Output Signals
v_fft_out = abs(fftshift(fft(v_bpsk_out(find(p_bpsk_in > -5,1),:),window_size))).^2;
npsd_out = v_fft_out/max(v_fft_out);

v_fft_out_2 = abs(fftshift(fft(v_bpsk_out(find(p_bpsk_in > -10,1),:),window_size))).^2;
npsd_out_2 = v_fft_out_2/max(v_fft_out_2);

v_fft_out_3 = abs(fftshift(fft(v_bpsk_out(find(p_bpsk_in > -15,1),:),window_size))).^2;
npsd_out_3 = v_fft_out_3/max(v_fft_out_3);

% RRC-shaped Baseband
[~,number_of_samples]= size(w_bpsk);
window_size = number_of_samples;
f_rrc = ((0: window_size - 1)-window_size/2)*(f_DAC/window_size);

% Input Signal
w_fft_in = abs(fftshift(fft(w_bpsk(2,:),window_size))).^2;
npsd_rrc_in = w_fft_in/max(w_fft_in);

% Output Signals
w_fft_out = abs(fftshift(fft(w_bpsk_out(find(p_bpsk_rrc_in > -5,1),:),window_size))).^2;
npsd_rrc_out = w_fft_out/max(w_fft_out);

w_fft_out_2 = abs(fftshift(fft(w_bpsk_out(find(p_bpsk_rrc_in > -10,1),:),window_size))).^2;
npsd_rrc_out_2 = w_fft_out_2/max(w_fft_out_2);

w_fft_out_3 = abs(fftshift(fft(w_bpsk_out(find(p_bpsk_rrc_in > -15,1),:),window_size))).^2;
npsd_rrc_out_3 = w_fft_out_3/max(w_fft_out_3);

disp('Done!')
% Moving Average Filter to smooth graphs
wndwSize = 5;
h = ones(1,wndwSize)/wndwSize;
wndwSize_rcc = 25;
h_rrc = ones(1,wndwSize_rcc)/wndwSize_rcc;
h = 1;
h_rrc = 1;

% Plots
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,1,1);
plot(f, filter(h,1,10*log10(npsd_in))); 
title('BPSK Input Signal Spectrum');
xlim([0 45e6])
ylim([-50 0])
grid on

subplot(2,1,2);
plot(f, filter(h,1,10*log10(npsd_out)),...
     f, filter(h,1,10*log10(npsd_out_2)),'g',...
     f, filter(h,1,10*log10(npsd_out_3)),'r'); 
title('BPSK Output Signal Spectrum');
legend('P_{in} = -5dBm','P_{in} = -10dBm','P_{in} = -15dBm')
xlim([0 45e6])
ylim([-50 0])
grid on

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,1,1);
plot(f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_in))); 
title('RRC-Shaped BPSK Input Signal Spectrum');
xlim([0 45e6])
ylim([-50 0])
grid on

subplot(2,1,2);
plot(f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out)),...
     f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out_2)),'g',...
     f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out_3)),'r'); 
title('RRC-Shaped BPSK Output Signal Spectrum');
legend('P_{in} = -5dBm','P_{in} = -10dBm','P_{in} = -15dBm')
xlim([0 45e6])
ylim([-50 0])
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QPSK
disp('Calculate QPSK spectrum...')
% Unshaped Baseband
[~,number_of_samples]= size(v_qpsk);
window_size = number_of_samples;
f = ((0: window_size - 1)-window_size/2)*(f_DAC/window_size);

% Input Signal
v_fft_in = abs(fftshift(fft(v_qpsk(2,:),window_size))).^2;
npsd_in = v_fft_in/max(v_fft_in);

% Output Signals
v_fft_out = abs(fftshift(fft(v_qpsk_out(find(p_qpsk_in > -5,1),:),window_size))).^2;
npsd_out = v_fft_out/max(v_fft_out);

v_fft_out_2 = abs(fftshift(fft(v_qpsk_out(find(p_qpsk_in > -10,1),:),window_size))).^2;
npsd_out_2 = v_fft_out_2/max(v_fft_out_2);

v_fft_out_3 = abs(fftshift(fft(v_qpsk_out(find(p_qpsk_in > -15,1),:),window_size))).^2;
npsd_out_3 = v_fft_out_3/max(v_fft_out_3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RRC-shaped Baseband
[~,number_of_samples]= size(w_qpsk);
window_size = number_of_samples;
f_rrc = ((0: window_size - 1)-window_size/2)*(f_DAC/window_size);

% Input Signal
w_fft_in = abs(fftshift(fft(w_qpsk(2,:),window_size))).^2;
npsd_rrc_in = w_fft_in/max(w_fft_in);

% Output Signals
w_fft_out = abs(fftshift(fft(w_qpsk_out(find(p_qpsk_rrc_in > -5,1),:),window_size))).^2;
npsd_rrc_out = w_fft_out/max(w_fft_out);

w_fft_out_2 = abs(fftshift(fft(w_qpsk_out(find(p_qpsk_rrc_in > -10,1),:),window_size))).^2;
npsd_rrc_out_2 = w_fft_out_2/max(w_fft_out_2);

w_fft_out_3 = abs(fftshift(fft(w_qpsk_out(find(p_qpsk_rrc_in > -15,1),:),window_size))).^2;
npsd_rrc_out_3 = w_fft_out_3/max(w_fft_out_3);

disp('Done!')
% Moving Average Filter to smooth graphs
wndwSize = 5;
h = ones(1,wndwSize)/wndwSize;
wndwSize_rcc = 25;
h_rrc = ones(1,wndwSize_rcc)/wndwSize_rcc;
h = 1;
h_rrc = 1;

% Plots
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,1,1);
plot(f, filter(h,1,10*log10(npsd_in))); 
title('QPSK Input Signal Spectrum');
xlim([0 45e6])
ylim([-50 0])
grid on

subplot(2,1,2);
plot(f, filter(h,1,10*log10(npsd_out)),...
     f, filter(h,1,10*log10(npsd_out_2)),'g',...
     f, filter(h,1,10*log10(npsd_out_3)),'r'); 
title('QPSK Output Signal Spectrum');
legend('P_{in} = -5dBm','P_{in} = -10dBm','P_{in} = -15dBm')
xlim([0 45e6])
ylim([-50 0])
grid on

figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,1,1);
plot(f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_in))); 
title('RRC-Shaped QPSK Input Signal Spectrum');
xlim([0 45e6])
ylim([-50 0])
grid on

subplot(2,1,2);
plot(f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out)),...
     f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out_2)),'g',...
     f_rrc, filter(h_rrc,1,10*log10(npsd_rrc_out_3)),'r'); 
title('RRC-Shaped QPSK Output Signal Spectrum');
legend('P_{in} = -5dBm','P_{in} = -10dBm','P_{in} = -15dBm')
xlim([0 45e6])
ylim([-50 0])
grid on









