%starter code for project 1: linear regression
%pattern recognition, CSE583/EE552
%{
    Name:Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: (This script generates noisy observations from the ground truth curve).
%}

npts = 50; % number of sample points -- change this number when you want to vary the sample size
x = linspace(1,4*pi,npts);
y = sin(.5*x);

% define the noise model
nmu = 0;
sigma = 0.3;
noise = nmu+sigma.*randn(1,npts); % generate npts number of samples from the N(nmu,nsigma^2)
t  = y + noise; % noisy observation

% save the data points 
save /Users/anishphule/Downloads/Project_1/data/noisy_data.mat x y t sigma

