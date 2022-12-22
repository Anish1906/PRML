% Start code for Project 1-Part 2: Classification
% CSE583/EE552 PRML
% TA: Shimian Zhang, Jan 2022
% TA: Addison Petro, Jan 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: (This script does the classification for the Taiji dataset).
%}

close all;
clear all;
addpath visualization;
mkdir result;

%%  An example of Linear Discriminant Classification

%   Choose which dataset to use (choices: wallpaper, taiji)
dataset = 'taiji';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
K = length(countcats(test_labels));

featureVector = train_featureVector.';

class = [1767, 1066, 2132, 1066, 1066, 2132, 1066, 1066];

sum0 = 0;sum1 = 0;sum2 = 0;sum3 = 0;sum5 = 0;sum6 = 0;sum7 = 0;sum9 = 0;
sum = 0;

%calculating mean for each class k
for k = 1:1:64
    for j = 1:1:11361
        if train_labels(j) == '0'
            sum0 = sum0 + featureVector(k,j);
            mk(k,1) = sum0/1767;
        elseif train_labels(j) == '1'
            sum1 = sum1 + featureVector(k,j);
            mk(k,2) = sum1/1066;
        elseif train_labels(j) == '2'
            sum2 = sum2 + featureVector(k,j);
            mk(k,3) = sum2/2132;
        elseif train_labels(j) == '3'
            sum3 = sum3 + featureVector(k,j);
            mk(k,4) = sum3/1066;
        elseif train_labels(j) == '5'
            sum5 = sum5 + featureVector(k,j);
            mk(k,5) = sum5/1066;
        elseif train_labels(j) == '6'
            sum6 = sum6 + featureVector(k,j);
            mk(k,6) = sum1/2132;
        elseif train_labels(j) == '7'
            sum7 = sum7 + featureVector(k,j);
            mk(k,7) = sum1/1066;
        elseif train_labels(j) == '9'
            sum9 = sum9 + featureVector(k,j);
            mk(k,8) = sum9/1066;
        end
    end
end

%calculating overall mean
for k = 1:1:64
    for i=1:1:8
        sum = sum + mk(k,i);
    end
    m(k) = sum/8;
    sum = 0;
end

mt = m';
m_diff = zeros(64,8);

for i=1:1:8
    m_diff(:,i) = mk(:,i) - mt;
end 

SB = zeros(64,64);
sb0 = zeros(64,64);

%Inter-class covariance matrix
for i = 1:1:8
    sb0 = class(i)*(m_diff(:,i)).*((m_diff(:,i)'));
    SB = SB + sb0;
end

m_diff2 = zeros(64,11361);
sk0 = zeros(64,64);
Sk = zeros(64,64);
SW = zeros(64,64);

for k = 1:1:64
    for j = 1:1:11361
        if train_labels(j) == '0'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,1);
        elseif train_labels(j) == '1'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,2);
        elseif train_labels(j) == '2'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,3);
        elseif train_labels(j) == '3'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,4);
        elseif train_labels(j) == '5'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,5);
        elseif train_labels(j) == '6'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,6);
        elseif train_labels(j) == '7'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,7);
        elseif train_labels(j) == '9'
            m_diff2(k,j) =  featureVector(k,j) - mk(k,8);
        end
    end
end

%between class covariance matrix
for i = 1:1:11361
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
w = w(:,1:4);

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
featureB = 2;
feature_idx = [featureA,featureB];
visu_train_featureVector = Y(:,feature_idx);
visu_test_featureVector = Y_test(:,feature_idx);
MdlLinear2 = fitcdiscr(visu_train_featureVector,train_labels);
figure()
h = visuBoundFill(MdlLinear2,visu_test_featureVector,test_labels,1,2);
title('{\bf Classification Area}')
exportgraphics(gcf, sprintf('result/%s_classification.png', dataset));


