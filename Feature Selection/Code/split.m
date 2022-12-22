% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Splits the data into training and testing data
%}
function [TrainMat, TrainLabel, TestMat, TestLabel] = split(data, label, sub_info, i)
% Input: 
%   data - a NxM matrix that contains the full list of features of Taiji data. 
% N is the number of frames and M is the dimension of the feature. 
%   label - a N vector that contains form label of each frame.
%   sub_info - a Nx2 matrix that contains [subject_id, take_id] of each
%   frame
%   i - a selected subject id to be used for testing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Among 10 subjects, choose one subject as test subject, and other 9
% subjects are trainins subjects.

% get test data 
test_list = sub_info(:,1) == i; % i-th subject is for the test.
train_list = sub_info(:,1) ~= i; % others are for training

TrainMat = data(train_list,:);
TrainLabel = label(train_list,:);

TestMat = data(test_list,:);
TestLabel = label(test_list,:);

end