function  ExampleComputing( inputnumber )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%% This program is to implement the GM-estimator using projection statistics for power system state estimation %%%%%%
%%%%% The test systems include IEEE 14-bus, 30-bus and 118-bus systems. %%%%%
%%%%% Only SCADA measrements are used for steady-state estimation
%%%%% Author: Junbo Zhao, Lamine Mili
%%%%% 09/16/2015
%%%%% Email: zjunbo@vt.edu, junbobzhao@gmail.com and lmili@vt.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% When you use this code for your research, we would appreciate if you cite the following papers%%
%% [R1] L. Mili, M. Cheniae, N. Vichare, and P. Rousseeuw, ``Robust state estimation based on projection statistics," IEEE Trans. Power Syst, vol. 11, no. 2, p. 1118--1127, 1996.
%% [R2] J.B Zhao, G. X. Zhang, Massimo La Scala, " A Two-stage Robust Power System State Estimation Method with Unknown Measurement Noise", IEEE PES General Meeting, 2016.
%%%%%% Following is the main program %%%%%%%
%%%%%%-----------------------------------%%%%%%%
%nbus represents the number of bus system
% close all;
% clear;
% clc;
nbus = inputnumber;
%nbus=5 % for IEEE 5 bus system
%nbus = 14 %for IEEE 14 bus system
%nbus = 30 %for IEEE 30 bus system
%nbus = 118 %for IEEE 118 bus system
ybus = ybusfunc(nbus); % Get YBus..
zdata = zconv(nbus); % Get Conventional Measurement data..
[bsh g b] = line_mat_func(nbus); % Get conductance and susceptance matrix 
type = zdata(:,2); 
% Type of measurement,
% type =1 voltage magnitude p.u
% type =2 Voltage phase angle in degree
% type =3 Real power injections
% type =4 Reactive power injection
% type =5 Real power flow 
% type =6 Reactive power flow 
z = zdata(:,3); % Measurement values
Z=z;% for ploting figures
fbus = zdata(:,4); % From bus
tbus = zdata(:,5); % To bus
Ri = diag(zdata(:,6)); % Measurement Error Covariance matrix
e = ones(nbus,1); % Initialize the real part of bus voltages
f = zeros(nbus,1);% Initialize the imaginary part of bus voltages
E = [f;e];  % State Vector comprising of imaginary and real part of voltage
G = real(ybus);
B = imag(ybus);
ei = find(type == 1); % Index of voltage magnitude measurements..
fi = find(type == 2); % Index of voltage angle measurements..
ppi = find(type == 3); % Index of real power injection measurements..
qi = find(type == 4); % Index of reactive power injection measurements..
pf = find(type == 5); % Index of real power flow measurements..
qf = find(type == 6); % Index of reactive power flow measurements..
Vm=z(ei);
Thm=z(fi);
z(ei)=Vm.*cosd(Thm); % converting voltage from polar to Cartesian
z(fi)=Vm.*sind(Thm);
nei = length(ei); % Number of Voltage measurements(real)
nfi = length(fi); % Number of Voltage measurements(imaginary)
npi = length(ppi); % Number of Real Power Injection measurements..
nqi = length(qi); % Number of Reactive Power Injection measurements..
npf = length(pf); % Number of Real Power Flow measurements..
nqf = length(qf); % Number of Reactive Power Flow measurements..
nm=nei+nfi+npi+nqi+npf+nqf; % total number of measurements
% robust parameters
tol=1;
maxiter=30;% maximal iteration for iteratively reweighted least squares (IRLS) algorithm
c=1.5; % for Huber-estimator
bm=mad_factor(nm); % correction factor to achieve unbiasness under Gaussian measurement noise
%%%%%%% GM-estimator%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% flat initialization
    iter=1;
    s=1;
%% For the GM-estimator to be able to handle two conforming outliers located on the same bus
%% the local redundancy must be large enough
%% add outliers %% 

%% For IEEE 14-bus system, injections of buses 4 (z(4,1) and z(11,1)), 8(z(5,1) and z(12,1)), 10(z(6,1) and z(13,1)), 14(z(9,1) and z(16,1))
%% and power flow P_{4-7} (z(20,1) and z(28,1)) are critical measurements
%% for any method, if there is not enough local measurement redundancy, the bad data in critical measurements can not be handled
% z(18,1)=1; % power flow P_{2-3} for IEEE 14-bus--can handle
% z(7,1)=-0.1; % power injection P_11 for IEEE 14-bus--can handle
% z(14,1)=-0.1; % power injection Q_11 for IEEE 14-bus--can handle
% You can keep testing as long as no bad critical measurement is added, the
% GM-estimator can bound the influence of the outlier

%% For IEEE 30-bus system, injections of bues 11 (z(9,1) and z(24,1)), bus 12 (z(10,1) and z(25,1)), bus 24 (z(15,1) and z(30,1)), 
%% bus 27 (z(16,1) and z(31,1)), bus 30(z(17,1) and z(32,1)), real power injections P_15, P_17 are also critical measurements
%% power flow along line 24-23 (corresponding to measurements z(52,1) and z(75,1)), 25-26 (z(53,1) and z(76,1)), 30-27 (z(55,1) and z(78,1))
%% add outliers
%  z(7,1)=-0.1; % power injection P_8--can handle
%  z(3,1)=0.1; % power injection P_2--corresponding to the bad leverage points--can handle
%  z(14,1)=-0.1; % power injection P_21--corresponding to the bad leverage points--can handle
%  z(18,1)=1; % power injection Q_2--can handle
%  z(8,1)=0.1; % power injection P_10--can handle
%% construct the multiple interacting and conforming bad data
%% case 1
%% z(32,1)=0.1; % power injection P_30--corresponding to the bad leverage points--can not hand since it is the critical measurement
%% z(55,1)=0.1; % power injection P_{30-27}--corresponding to the bad leverage points--can not hand since it is the critical measurement
%% case 2  multiple interacting and conforming bad data
%  z(13,1)=0.1; % power flow P_19--corresponding to the bad leverage points--can handle
%  z(48,1)=0.1; % power flow P_{19-20}--corresponding to the bad leverage points--can handle
%  z(71,1)=0.1; % power flow Q_{19-20}--corresponding to the bad leverage points--can handle
%% case 3
%  z(6,1)=0.2; % power injection P_11---corresponding to zero power injection-can handle
%%
%  z(48,1)=0.1; % power flow P_{19-20}--corresponding to the bad leverage points--can handle
%  z(71,1)=0.1; % power flow Q_{19-20}--corresponding to the bad leverage points--can handle
%% when bad data occur in critical measurements, GM-estimator can not handle them only when alternative redundancy is introduced
%% Actually, there is no method can handle bad critical measurements if no alternative redundancy is introduced
% z(9,1)=0.1; % power flow P_{19-20}--corresponding to the bad critical measurement--can not handle it but it can bound its influence a little bit
% z(15,1)=0.1; % power injection P_15--corresponding to the bad critical measurement--can not handle it but it can bound its influence a little bit
%% You can keep testing as long as no bad critical measurement is added, the
%% GM-estimator can bound the influence of the outlier

%% For IEEE 118-bus system, injections of bues 34 (corresponding to measurements z(10,1) and z(49,1)), 79 (z(25,1) and z(64,1)), 113 (z(39,1) and z(78,1)),
%% bus 114 (z(40,1) and z(79,1)) 
%% bus 117 (z(42,1) and z(81,1)), P_{1-2}(z(83,1)),P_{3-5}(z(84,1)),P_{3-1}(z(85,1)) are critical power injection measurements
%% power flow along line 5-6 (corresponding to measurements z(88,1) and z(199,1)), line 7-12 (z(89,1) and z(200,1)), line 8-9 z(90,1) and z(201,1)), 9-10(z(91,1) and z(202,1)),
%% 14-12 (z(93,1) and z(204,1)), 29-31 (z(112,1) and z(223,1)),32-31 (z(118,1) and z(229,1)), 36-35 (z(120,1) and z(231,1)), 51-52 (z(134,1) and z(245,1)), 
%% 52-53 (z(135,1) and z(246,1)), 54-55 (z(136,1) and z(247,1)), 54-59 (z(137,1) and z(248,1)), 59-60 (z(141,1) and z(252,1)), 59-61 (z(142,1) and z(253,1)),
%% 61-64 (z(143,1) and z(254,1)), 63-59 (z(144,1) and z(255,1)),77-78 (z(158,1) and z(269,1)), 83-84 (z(162,1) and z(273,1)), 86-85 (z(165,1) and z(276,1)), 
%% 87-86 (z(166,1) and z(277,1)),96-95 (z(176,1) and z(287,1))
%% 96-97 (z(177,1) and z(288,1)), Q_{1-2} (z(194,1)),Q_{3-5}(z(195,1)),Q_{3-1}(z(196,1)) are critical power flow measurements
%% add outliers
%  z(6,1)=-0.1; % zero power injection P_6--corresponding to the bad leverage point-can handle
%  z(9,1)=-0.1; % zero power injection P_11--can handle
%% construct the multiple interacting and conforming bad data
%  z(12,1)=0.1; % power injection P_40--corresponding to the bad leverage points--can handle
%  z(51,1)=0.1; % power injection Q_40--corresponding to the bad leverage points--can handle
%%
%  z(14,1)=-0.1; % power injection P_44--corresponding to the bad leverage points--can handle
%  z(18,1)=1; % power injection P_49--can handle
%  z(193,1)=1; % power injection Q_{118-76}--can handle
%  z(80,1)=0.1; % power injection Q_116--corresponding to the bad leverage point--can handle
%  z(108,1)=0.1; % power flow P_{26-25}--can handle
%  z(202,1)=0.1; % power flow Q_{9-10}--corresponding to the bad leverage point--can handle
%% z(20,1)=0.3; % zero power injection Q_4--corresponding to the bad leverage point-can not handle--need to find the reason (might be the low local redundancy)
%% Calculate the measurements
h1 = e(fbus (ei),1);  %voltage measurement
h2 = f(fbus (fi),1);  %angle measurement
h3 = zeros(npi,1);  %real power injection
h4 = zeros(nqi,1);  %reactive power injection
h5 = zeros(npf,1);  %real power flow
h6 = zeros(nqf,1);  %reactive power flow
%Measurement function of power injection
for i = 1:npi
m = fbus(ppi(i));
for k = 1:nbus
% Real injection
h3(i)=h3(i)+(G(m,k)*(e(m)*e(k)+f(m)*f(k))+B(m,k)*(f(m)*e(k)-e(m)*f(k)));
% Reactive injection 
h4(i)=h4(i)+(G(m,k)*(f(m)*e(k)-e(m)*f(k))-B(m,k)*(e(m)*e(k)+f(m)*f(k)));
end
end
%Measurement function of power flow
for i = 1:npf
    m = fbus(pf(i));
    n = tbus(pf(i));
% Real injection
h5(i) =(e(m)^2 + f(m)^2)*g(m,n)-(g(m,n)*(e(m)*e(n)+f(m)*f(n))+b(m,n)*(f(m)*e(n)-e(m)*f(n)));
% Reactive injection 
h6(i) =-g(m,n)*(f(m)*e(n)-e(m)*f(n))+b(m,n)*(e(m)*e(n)+f(m)*f(n))-(e(m)^2 + f(m)^2)*(b(m,n)+bsh(m,n));
end
h = [h1; h2; h3; h4; h5; h6];
%% Calculate the Jacobian matrix
% Jacobian..
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 1: Derivative of voltage %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H11 = zeros(nei,nbus); % Derivative of e wrt e
H12 = zeros(nei,nbus); % Derivative of e wrt f
H21 = zeros(nfi,nbus); % Derivative of f wrt e
H22 = zeros(nfi,nbus); % Derivative of f wrt f
for k = 1:nei
H11(k,fbus(k)) = 1;
H22(k,fbus(n)) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 2: Derivative of Power injection %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
H31 = zeros(npi,nbus);  %Derivative of real power injection wrt e
H32 = zeros(npi,nbus);  %Derivative of real power injection wrt f
H41 = zeros(npi,nbus);  %Derivative of reactive power injection wrt e
H42 = zeros(npi,nbus);  %Derivative of reactive power injection wrt f
for i = 1:npi
m = fbus(ppi(i));
for k = 1:(nbus)
if k == m
for n = 1:nbus
H31(i,k) = H31(i,k) + (G(m,n)*e(n) - B(m,n)*f(n));
H32(i,k) = H32(i,k) + (G(m,n)*f(n) + B(m,n)*e(n));
H41(i,k) = H41(i,k) -G(m,n)*f(n) - B(m,n)*e(n);
H42(i,k) = H42(i,k) + (G(m,n)*e(n) - B(m,n)*f(n));
end
H31(i,k) = H31(i,k) + f(m)*B(m,m) + G(m,m)*e(m);
H32(i,k) = H32(i,k) - e(m)*B(m,m) + f(m)*G(m,m);
H41(i,k) = H41(i,k) + f(m)*G(m,m) - e(m)*B(m,m);
H42(i,k) = H42(i,k) - e(m)*G(m,m) - f(m)*B(m,m);
else
H31(i,k) = G(m,k)*e(m) + B(m,k)*f(m);
H32(i,k) =G(m,k)*f(m) - B(m,k)*e(m); 
H41(i,k) = (G(m,k)*f(m) - B(m,k)*e(m));
H42(i,k) = (-G(m,k)*e(m) - B(m,k)*f(m));
end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 3: Derivative of Power flow %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
H51 = zeros(npf,nbus);
H52 = zeros(npf,nbus);
H61 = zeros(nqf,nbus);
H62 = zeros(nqf,nbus);
for i = 1:npf
m = fbus(pf(i));
n = tbus(pf(i)); 
H51(i,m) = 2*e(m)*g(m,n) - g(m,n)*e(n) + b(m,n)*f(n); 
H51(i,n) = -g(m,n)*e(m) - b(m,n)*f(m);
H52(i,m) = 2*f(m)*g(m,n) - g(m,n)*f(n) - b(m,n)*e(n);
H52(i,n) = -g(m,n)*f(m) + b(m,n)*e(m); 
H61(i,m)=-2*e(m)*(b(m,n)+bsh(m,n))+g(m,n)*f(n)+b(m,n)*e(n);
H61(i,n) = -g(m,n)*f(m) + b(m,n)*e(m); 
H62(i,m)=-2*f(m)*(b(m,n)+bsh(m,n))-g(m,n)*e(n)+b(m,n)*f(n);
H62(i,n) = g(m,n)*e(m) + b(m,n)*f(m);
end
% Measurement Jacobian, H..
H = [H11 H12;
H21 H22;
H31 H32;
H41 H42;
H51 H52;
H61 H62]; 
%% Indentify leverage points (bad or good)
%% Calculate the corresponding weight
%% projection statistics--From the paper published by Dr. Mili on 1996
    PSi=PS_sparse(H);
    [m,n]=size(H);
for i=1:m
niu=sum(H(i,:)~=0);
cuttoff_PS(i,1)=testchi2(0.975,niu);
w(i,1)=min(1,(cuttoff_PS(i,1)./PSi(i))^2); %% downweight the outliers or leverage points
end
%%
%% finish the identifying of outliers
%% start to iterate using IRLS algorithm
%%
while(tol > 1e-6)
%Measurement Function, h
h1 = e(fbus (ei),1);  %voltage measurement
h2 = f(fbus (fi),1);  %angle measurement
h3 = zeros(npi,1);  %real power injection
h4 = zeros(nqi,1);  %reactive power injection
h5 = zeros(npf,1);  %real power flow
h6 = zeros(nqf,1);  %reactive power flow
%Measurement function of power injection
for i = 1:npi
m = fbus(ppi(i));
for k = 1:nbus
% Real injection
h3(i)=h3(i)+(G(m,k)*(e(m)*e(k)+f(m)*f(k))+B(m,k)*(f(m)*e(k)-e(m)*f(k)));
% Reactive injection 
h4(i)=h4(i)+(G(m,k)*(f(m)*e(k)-e(m)*f(k))-B(m,k)*(e(m)*e(k)+f(m)*f(k)));
end
end
%Measurement function of power flow
for i = 1:npf
    m = fbus(pf(i));
    n = tbus(pf(i));
% Real injection
h5(i) =(e(m)^2 + f(m)^2)*g(m,n)-(g(m,n)*(e(m)*e(n)+f(m)*f(n))+b(m,n)*(f(m)*e(n)-e(m)*f(n)));
% Reactive injection 
h6(i) =-g(m,n)*(f(m)*e(n)-e(m)*f(n))+b(m,n)*(e(m)*e(n)+f(m)*f(n))-(e(m)^2 + f(m)^2)*(b(m,n)+bsh(m,n));
end
h = [h1; h2; h3; h4; h5; h6];
%Residual matrix difference of measurement and the non linear 
%r = z - h; 
% Jacobian..
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 1: Derivative of voltage %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H11 = zeros(nei,nbus); % Derivative of e wrt e
H12 = zeros(nei,nbus); % Derivative of e wrt f
H21 = zeros(nfi,nbus); % Derivative of f wrt e
H22 = zeros(nfi,nbus); % Derivative of f wrt f
for k = 1:nei
H11(k,fbus(k)) = 1;
H22(k,fbus(n)) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 2: Derivative of Power injection %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
H31 = zeros(npi,nbus);  %Derivative of real power injection wrt e
H32 = zeros(npi,nbus);  %Derivative of real power injection wrt f
H41 = zeros(npi,nbus);  %Derivative of reactive power injection wrt e
H42 = zeros(npi,nbus);  %Derivative of reactive power injection wrt f
for i = 1:npi
m = fbus(ppi(i));
for k = 1:(nbus)
if k == m
for n = 1:nbus
H31(i,k) = H31(i,k) + (G(m,n)*e(n) - B(m,n)*f(n));
H32(i,k) = H32(i,k) + (G(m,n)*f(n) + B(m,n)*e(n));
H41(i,k) = H41(i,k) -G(m,n)*f(n) - B(m,n)*e(n);
H42(i,k) = H42(i,k) + (G(m,n)*e(n) - B(m,n)*f(n));
end
H31(i,k) = H31(i,k) + f(m)*B(m,m) + G(m,m)*e(m);
H32(i,k) = H32(i,k) - e(m)*B(m,m) + f(m)*G(m,m);
H41(i,k) = H41(i,k) + f(m)*G(m,m) - e(m)*B(m,m);
H42(i,k) = H42(i,k) - e(m)*G(m,m) - f(m)*B(m,m);
else
H31(i,k) = G(m,k)*e(m) + B(m,k)*f(m);
H32(i,k) =G(m,k)*f(m) - B(m,k)*e(m); 
H41(i,k) = (G(m,k)*f(m) - B(m,k)*e(m));
H42(i,k) = (-G(m,k)*e(m) - B(m,k)*f(m));
end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Jacobian Block 3: Derivative of Power flow %%%%%
%%%%% with respect to states %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
H51 = zeros(npf,nbus);
H52 = zeros(npf,nbus);
H61 = zeros(nqf,nbus);
H62 = zeros(nqf,nbus);
for i = 1:npf
m = fbus(pf(i));
n = tbus(pf(i)); 
H51(i,m) = 2*e(m)*g(m,n) - g(m,n)*e(n) + b(m,n)*f(n); 
H51(i,n) = -g(m,n)*e(m) - b(m,n)*f(m);
H52(i,m) = 2*f(m)*g(m,n) - g(m,n)*f(n) - b(m,n)*e(n);
H52(i,n) = -g(m,n)*f(m) + b(m,n)*e(m); 
H61(i,m)=-2*e(m)*(b(m,n)+bsh(m,n))+g(m,n)*f(n)+b(m,n)*e(n);
H61(i,n) = -g(m,n)*f(m) + b(m,n)*e(m); 
H62(i,m)=-2*f(m)*(b(m,n)+bsh(m,n))-g(m,n)*e(n)+b(m,n)*f(n);
H62(i,n) = g(m,n)*e(m) + b(m,n)*f(m);
end
% Measurement Jacobian, H..
H = [H11 H12;
H21 H22;
H31 H32;
H41 H42;
H51 H52;
H61 H62]; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%indentify and downweight
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%the leverage points
%     %projection statistics------------------------------------------------
%     PSi=PS_sparse(H);
%     [m,n]=size(H);
% for i=1:m
% niu=sum(H(i,:)~=0);
% cuttoff_PS(i,1)=chi2inv(0.975,niu);
% w(i,1)=min(1,(cuttoff_PS(i,1)./PSi(i))^2); %% downweight the outliers
% end
 %%%%%%%%%%%%%%%%%%%%% IRLS algorithm
 %% GM-estimator with PS
        ri =z-h;       
        for i=1:nm
           rsi(i)=ri(i)./(w(i,1)*sqrt(Ri(i,i))); % if the measurement noise is known and is following the Gaussian distribution
         %%  rsi(i)=ri(i)./(w(i).*s);  % with the robust scale estimation. s represents the unknown distribution of the measurement noise.
         %% That means there is no necessary to make the Gaussian distribution assumption
        end
%       rsi = ri./(wi.*s);
        for i=1:(nm)
            if abs(rsi(i))<=c
                 QQ(i,i)=1;
            else
                % QQ(i,i)=c/abs(rsi(i));
                 QQ(i,i)=c*sign(rsi(i))./rsi(i);
            end
        end
        %dE=inv(H'*QQ*H)*H'*QQ*ri; % the difference of the state vector at different iteration
        dE=inv(H'*inv(Ri)*QQ*H)*H'*inv(Ri)*QQ*ri; % the difference of the state vector at different iteration
        E=E+dE;
        iter=iter+1;
        e = E(1:nbus);
        f = E(nbus+1:end);
        s = 1.4826*bm*median(abs(ri)); % estimate the robust scale parameter
        tol=max(abs(dE));
end   

%displayout(E,'a'); % Displaying output in tabular form
f = E(nbus+1:end);
e = E(1:nbus);
v=e+1i*f;
V=abs(v);
Del=round(angle(v)*180/pi*100)/100;
disp('-------- State Estimation ------------------');
disp('--------------------------');
disp('| Bus |    V   |  Angle  | ');
disp('| No  |   pu   |  Degree | ');
disp('--------------------------');
for m = 1:nbus
    fprintf('%4g', m); fprintf('  %8.4f', V(m)); fprintf('   %8.4f', Del(m)); fprintf('\n');
end
disp('---------------------------------------------');
%% calculate the estimated value
%Measurement Function, h
h1 = V(fbus (ei),1);  %voltage measurement
h2 = Del(fbus (fi),1);  %angle measurement
h3 = zeros(npi,1);  %real power injection
h4 = zeros(nqi,1);  %reactive power injection
h5 = zeros(npf,1);  %real power flow
h6 = zeros(nqf,1);  %reactive power flow
%Measurement function of power injection
for i = 1:npi
m = fbus(ppi(i));
for k = 1:nbus
% Real injection
h3(i)=h3(i)+(G(m,k)*(e(m)*e(k)+f(m)*f(k))+B(m,k)*(f(m)*e(k)-e(m)*f(k)));
% Reactive injection 
h4(i)=h4(i)+(G(m,k)*(f(m)*e(k)-e(m)*f(k))-B(m,k)*(e(m)*e(k)+f(m)*f(k)));
end
end
%Measurement function of power flow
for i = 1:npf
    m = fbus(pf(i));
    n = tbus(pf(i));
% Real injection
h5(i) =(e(m)^2 + f(m)^2)*g(m,n)-(g(m,n)*(e(m)*e(n)+f(m)*f(n))+b(m,n)*(f(m)*e(n)-e(m)*f(n)));
% Reactive injection 
h6(i) =-g(m,n)*(f(m)*e(n)-e(m)*f(n))+b(m,n)*(e(m)*e(n)+f(m)*f(n))-(e(m)^2 + f(m)^2)*(b(m,n)+bsh(m,n));
end
%% note that the angle measurement should be converted to radians for measurement comparison
h = [h1; h2; h3; h4; h5; h6];
% %% % the estimated voltage and the true voltage magnitude in p.u.
% figure(1) 
% K=1:1:nbus;
 [Vtrue Angletrue]=IEEE_true_value(nbus); % true voltage magnitude
% plot(K,V,'r:*',K,Vtrue,'b--o','linewidth',1.5)
% title('Volatge Magnitude Comparision Result ')
% xlabel('Bus number')
% xlim([1 nbus])
% ylabel('Voltage in p.u')
% legend('Estimated Value','True Value',1)
% grid on
% %% % the estimated voltage angle and the true voltage angle in degree
% figure(2)
% j=1:1:nbus;
% plot(j,Del,'r:*',j,Angletrue,'b--o','linewidth',1.5)
% title('Voltage Angle Comparision Result')
% xlabel('Bus number')
% xlim([1 nbus])
% ylabel('Voltage angle in degree')
% legend('Estimated Value','True Value',1)
% grid on
% %% % the estimated and true measurement in degree
% figure(3)
% i=1:1:length(z);
% estimated_measurement=plot(i,Z,'b*',i,h,'r--o');
% set(estimated_measurement(1),'linewidth',1.5);
% set(estimated_measurement(2),'linewidth',1.5);
% title('Measurement Estimation Comparision Result')
% xlabel('Measurement number')
% xlim([1 length(z)])
% ylabel('Measurement value')
% legend('True Value','Estimated Value',1)
for i=1:nbus
voltage_error(i)=norm((Vtrue(i)-V(i)),inf)./abs(Vtrue(i));
angle_error(i)=norm((Angletrue(i)-Del(i)),inf)./abs(Angletrue(i));
end
Max_voltage_estimation_error=max(voltage_error)
Max_angle_estimation_error=max(angle_error)

end

