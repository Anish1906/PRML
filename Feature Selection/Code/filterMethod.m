% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: Filter method function
%}
function [indices, scores] = filterMethod(train_data, train_labels)
%% input
%{
trainData - NxM matrix that contains the full list of features of training 
data. N is the number of training samples and M is the dimension of the 
feature.
trainLabels - a Nx1 vector of the class labels for the training data
%}        
%% output:
%{
indices - a 1xM matrix sorted such that entry 1 contains the index of the
feature with the highest score, entry 2 contains the index of the feature
with the second highest score, etc.
scores - a 1xM matrix array where each entry is the corresponding feature's
score from the filtering method.
%}

%% Implementation,
% train_data = trainData;
% train_label = trainLabels;

%Sorting the labels and the data for classification
[sort_labels, sortIndex] = sort(train_labels,'ascend');
train_data1 = train_data(sortIndex,:);

train_size = size(train_data);
cnt = countlabels(categorical(sort_labels));
class = cnt(:,1);
class_size = size(class);
class = table2array(class);
count = cnt(:,2);
count = table2array(count);

%variance for each feature for dataset
varSF = var(train_data1, 0, 1);
class_var = zeros(44, 1961);

%creating dictionary to store particular class data
class_dict = containers.Map;

adder = 0;
for i=1:1:44
    class_dict(int2str(i-1)) = train_data1((adder+1):(adder + count(i)),:);
    adder = adder + count(i);
end

%claculating average variance of all classes for each feature
for i=1:1:44
    class_var(i,:) = var(class_dict(int2str(i-1)), 0, 1);
end

class_varsum = sum(class_var,1);

class_meanvar = class_varsum./44;

%final Variance ratio
VR = varSF./class_meanvar;
VR(isnan(VR))=0;

%Filtering by Sorting out highest to lowest VR values
[~, sortVR] = sort(VR,'descend');
VR1 = VR(:,sortVR);

indices = sortVR;
scores = VR;
% TODO: REPLACE DUMMY IMPLEMENTATION