% Start code for Project 1-Part 2: Classification
% CSE583/EE552 PRML
% TA: Shimian Zhang, Jan 2022
% TA: Addison Petro, Jan 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: (This script does the classification for the Wallpaper dataset).
%}

close all;
clear all;
addpath visualization;
mkdir result;

%%  An example of Linear Discriminant Classification

%   Choose which dataset to use (choices: wallpaper, taiji)
dataset = 'wallpaper';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
K = length(countcats(test_labels));

featureVector = train_featureVector.';

sum = 0;

%calculating mean for each class k
for i = 1:1:17
    for k = 1:1:500
        for j = ((i-1)*100+1):1:(i*100)
            sum = sum + featureVector(k,j);
        end
    mk(k,i) = sum/100;
    sum = 0;
    end
end
sum = 0;

%calculating overall mean
for k = 1:1:500
    for i=1:1:17
        sum = sum + mk(k,i);
    end
    m(k) = sum/17;
    sum = 0;
end
mt = m';
m_diff = zeros(500,17);

for i=1:1:17
    m_diff(:,i) = mk(:,i) - mt;
end 

SB = zeros(500,500);
sb0 = zeros(500,500);

%Inter-class covariance matrix
for i = 1:1:17
    sb0 = 100*(m_diff(:,i)).*((m_diff(:,i)'));
    SB = SB + sb0;
end

m_diff2 = zeros(500,1700);
sk0 = zeros(500,500);
Sk = zeros(500,500);
SW = zeros(500,500);

for i = 1:1:17
    for j = ((i-1)*100+1):1:i*100
        m_diff2(:,j) =  featureVector(:,j) - mk(:,i);
    end
end

%between class covariance matrix
for i = 1:1:1700
    sk0 = (m_diff2(:,i)).*((m_diff2(:,i)'));
    SW = SW + sk0;
end

SWi = inv(SW);
%weight values are given by eigen values and eigenvectors of Fisher
%criterion
[w, l] = eig(SWi*SB);

L = diag(l);
[L, Order]=sort(L,'descend');
w=w(:,Order);
w = w(:,1:9);

%applying weights to project data
Y=train_featureVector*w;
Y_test = test_featureVector*w;

%%  Classify the data and show statistics
%   This example is using Matlab's inbuilt Classifier. You will need first
%   implement Fisher Project on the data, and then do the classification.
%   Fit discriminant analysis classifier
%   https://www.mathworks.com/help/stats/fitcdiscr.html

MdlLinear = fitcdiscr(Y,train_labels);

%   Find the training accurracy 
train_pred = predict(MdlLinear,Y);

%   Create confusion matrix
train_ConfMat = confusionmat(train_labels,train_pred);
train_ConfMat = train_ConfMat./(meshgrid(countcats(train_labels))')
%   mean group accuracy and std
train_acc = mean(diag(train_ConfMat))
train_std = std(diag(train_ConfMat))

%   Find the testing accurracy 
test_pred = predict(MdlLinear,Y_test);

%   Create confusion matrix
test_ConfMat = confusionmat(test_labels,test_pred);
test_ConfMat = test_ConfMat./(meshgrid(countcats(test_labels))')
%   mean group accuracy and std
test_acc = mean(diag(test_ConfMat))
test_std = std(diag(test_ConfMat))

%%  Display the confusion matrix
figure()
draw_cm(train_ConfMat,categories(train_labels),K);
title('{\bf Train Confusion Matrix}')
exportgraphics(gcf, sprintf('result/%s_train_conf.png', dataset));
figure()
draw_cm(test_ConfMat,categories(test_labels),K);
title('{\bf Test Confusion Matrix}')
exportgraphics(gcf, sprintf('result/%s_test_conf.png', dataset));

%%  Display the classified regions of two of the feature dimensions  
%   To display the classified regions, you will need to project the
%   features to 2-Dimensions.
%   Here is a visualization example of selecting 2 features to retrain a
%   Classifier. (Notice that we are not projecting the features in this
%   example)

featureA = 1;
featureB = 7;
feature_idx = [featureA,featureB];
visu_train_featureVector = Y(:,feature_idx);
visu_test_featureVector = Y_test(:,feature_idx);
MdlLinear2 = fitcdiscr(visu_train_featureVector,train_labels);
figure()
h = visuBoundFill(MdlLinear2,visu_test_featureVector,test_labels,1,2);
title('{\bf Classification Area}')
exportgraphics(gcf, sprintf('result/%s_classification.png', dataset));


