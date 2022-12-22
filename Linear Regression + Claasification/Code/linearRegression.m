% Start code for Project 1-Part 1: linear regression
% CSE583/EE552 PRML
% TA: Shimian Zhang, Jan 2022
% TA: Addison Petro, Jan 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: 952051694
    Description: (This script does linear regression through ML and MAP methods).
%}

addpath visualization
mkdir result;

%% Generate noisy observations from the ground true curve
% load the data points
load /Users/anishphule/Downloads/Project_1/data/noisy_data.mat

%% Plot the ground truth curve of generated data
figure()
hold on;
% plot curve with red shaded region spans one standard deviation
shadedErrorBar(x,y,sigma.*ones(1,length(x))); 
% plot the noisy observations
plot(x,t,'ro','MarkerSize',8,'LineWidth',1.5);

hold off; 
% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')

% Save the image
exportgraphics(gcf, 'result/sample_curve.png');


%% Start your linear regression here

%we start building our ML equation first
%generating w*

%Design matrix for calculating w*
%X = cat(1, ones(1,length(x))); %M = 0
X = cat(1, ones(1,length(x)),x); %M = 1
%X = cat(1, ones(1,length(x)),x, x.^2, x.^3); %M = 3
%X = cat(1, ones(1,length(x)),x, x.^2, x.^3, x.^4, x.^5, x.^6); %M = 6
%X = cat(1, ones(1,length(x)),x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8,x.^9); %M = 9

%terms like X transpose, and inverse
Xt = X';
T = t';
H = X*Xt;
H_inv = inv(H);

%ML estimator
W_pre = H_inv*X;
W = W_pre*T;
Wt = W';
N = 50;

%Estimating 
Y = Wt*X;

figure()
hold on;
% plot curve with red shaded region spans one standard deviation
shadedErrorBar(x,Y,sigma.*ones(1,length(x)),[],1); 
% plot the noisy observations
plot(x,t,'ro','MarkerSize',8,'LineWidth',1.5);
plot(x,Y,'r', 'Linewidth', 2);
%plot(x,y,'LineWidth',1.5);

hold off; 
% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')

exportgraphics(gcf, 'result/MLE_curve.png');

%we start building our MAP
%generating w*_map

%alpha and beta values
alpha = 0.005;
beta = 11.1;

%lambda value
lambda = alpha/beta;
%Identity matrix
%I = eye(1);
I = eye(2);
%I = eye(4);
%I = eye(7);
%I = eye(10);

%estimator with prior distribution
J = (H + lambda*I);
J_inv = inv(J);
Wmap_pre = J_inv*X;
Wmap = Wmap_pre*T;
Wtmap = Wmap';

%Estimating
Ymap = Wtmap*X;

figure()
hold on;
% plot curve with red shaded region spans one standard deviation
%shadedErrorBar(x,Y,sigma.*ones(1,length(x)));
shadedErrorBar(x,Ymap,sigma.*ones(1,length(x)),'-b', 1); 
% plot the noisy observations
plot(x,t,'ro','MarkerSize',8,'LineWidth',1.5);
plot(x,Ymap,'b', 'Linewidth', 2);

hold off; 
% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')

exportgraphics(gcf, 'result/MAP_curve.png');

%Plotting both MLE and MAP in one plot
figure()
hold on;
% plot curve with red shaded region spans one standard deviation
%shadedErrorBar(x,Y,sigma.*ones(1,length(x)));
shadedErrorBar(x,Ymap,sigma.*ones(1,length(x)),'-b', 1); 
shadedErrorBar(x,Y,sigma.*ones(1,length(x)),'-r', 1); 
% plot the noisy observations
plot(x,t,'ro','MarkerSize',8,'LineWidth',1.5);
plot(x,Ymap,'b', 'Linewidth', 2);
plot(x,Y,'r', 'Linewidth', 2);

hold off; 
% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')

exportgraphics(gcf, 'result/Comparison_curve.png');