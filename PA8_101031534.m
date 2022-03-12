clc; 
close all; 
clear all;

%Jinseng Vanderkloot 
%101031534

Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;
V = linspace(-1.95, 0.7, 200);
I = Is.*(exp(V.*1.2/0.025)-1)+ Gp.*V- Ib.*(exp(-(V+Vb).*1.2/0.025)-1);
I_noise = I + I.*0.2.*rand(1,200);

%Plot Diode 
figure('name', 'Diode equation');
subplot(2,1,1); hold on;
plot(V,I);
plot(V,I_noise);
title('Diode equation');
xlabel('V'), ylabel('Id (A)');
legend('Id Ideal', 'Id with Noise');

subplot(2,1,2);
semilogy(V,I);
title('Diode equation Semilog');
xlabel('V'), ylabel('Id (A)');
legend('Id Ideal', 'Id with Noise');

%Polynimail Fitting 

EQ_4th =polyfit(V,I,4);
I_4th = polyval(EQ_4th, V);

EQ_8th =polyfit(V,I,8);
I_8th = polyval(EQ_8th, V);

figure('name', 'Diode equation Polynomial Fitting');
subplot(2,1,1); hold on;
plot(V,I);
plot(V,I_4th);
plot(V,I_8th);
title('Diode equation');
xlabel('V'), ylabel('Id (A)');
legend('Id Ideal', '4th degree Id', '8th degree Id');

subplot(2,1,2); hold on;
semilogy(V,I);
semilogy(V,I_4th);
semilogy(V,I_8th);
title('Diode equation Semilog');
xlabel('V'), ylabel('Id (A)');
legend('Id Ideal', '4th degree Id', '8th degree Id');

% Nonlinear curve fitting
fo = fittype('A.*(exp(1.2*x/25e-3)-1)+0.1.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V', I', fo);
If = ff(V);
ffn = fit(V', I_noise', fo);
Ifn = ffn(V);

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(V', I', fo2);
If2 = ff2(V);
ff2n = fit(V', I_noise', fo2);
If2n = ff2n(V);

fo3 = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(V', I', fo3);
If3 = ff3(V);
ff3n = fit(V', I_noise', fo3);
If3n = ff3n(V);

figure('name', 'Nonlinear curve fitting');
hold on;
subplot(2,1,1); hold on;
plot(V,I);
plot(V, If);
plot(V, If2);
plot(V, If3);
title('Non-Linear EQ no noise ');
xlabel('V'), ylabel('Id (A)');
legend('Id','Id 2Var', 'Id 3Var', 'Id 4Var');

subplot(2,1,2); hold on;
plot(V,I);
plot(V, Ifn);
plot(V, If2n);
plot(V, If3n);
title('Non-Linear EQ with noise');
xlabel('V'), ylabel('Id (A)');
legend('Id', 'Id 2Var', 'Id 3Var', 'Id 4Var');

%With more variables missing, the harder it is for the curve to match the
%original data. This is also not consistent. Somtimes the 4 Variable
%equation is able to match original data perfectly, other times it it
%completely wrong. The addition of noise causes the equations to fit less
%accurately then without noise when both work 

%Using Neural Network
inputs = V;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;

inputs = V;
targets = I_noise;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn_noise = outputs;

figure('name', 'Neural Network Fitting'); hold on;
plot(V,I); plot(V, Inn);plot(V, Inn_noise);
xlabel('V'), ylabel('Id (A)');
legend('Id', 'NN Id', 'NN Id Noise');

%The neural network learned from the real data and data with noise 
%converged to match the data almost quite accurately 

