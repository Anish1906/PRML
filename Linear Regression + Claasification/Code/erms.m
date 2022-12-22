%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: (This script calculates and plots the Erms and lnlambda values).
%}

% load the data points
load /Users/anishphule/Downloads/Project_1/data/noisy_data.mat

%Design matrix for calculating w*
%X = cat(1, ones(1,length(x)),x); %M = 1
X = cat(1, ones(1,length(x)),x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8, x.^9); %M = 9

%terms like X transpose, and inverse
N = 50;
Xt = X';
T = t';
H = X*Xt;
H_inv = inv(H);

%alpha and beta values
%beta = 11.1;
%alpha = lambda*beta;

%Identity matrix
%I = eye(2);
I = eye(10);

%estimator with prior distribution
j = 1;

%finding error function value
for i = -30:1:0
    %lambda values
    lnlambda = i;
    J = (H + exp(lnlambda)*I);
    J_inv = inv(J);
    Wmap_pre = J_inv*X;
    Wmap = Wmap_pre*T;
    Wtmap = Wmap';
    
    Kt = (T - Xt*H_inv*X*T);
    K = Kt';
    %Error function
    Ew = 0.5*(K*Kt) + 0.5*(exp(lnlambda))*Wtmap*Wmap;
    %finding Erms value
    Erms(j) = sqrt(2*Ew/N);
    x_axis(j) = i;
    j = j+1;
end

%plotting
plot(x_axis, Erms,'b', 'Linewidth', 2)
xlabel('lnlambda')
ylabel('Erms values')
ax = gca;
ax.FontSize = 15;
exportgraphics(gcf, 'result/Erms.png');