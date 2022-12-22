% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Forward selection function
%}
function fs_indices = forwardSelection(train_data, train_labels, filter_select_count)
clear feature_index final_index score_new score_old f train_acc score_track count data_aug  MdlLinear train_pred train_ConfMat max_index 
%% input
%{
trainData - NxM matrix that contains the full list of features of training 
data. N is the number of training samples and M is the dimension of the 
feature.
trainLabels - a Nx1 vector of the class labels for the training data
%}        
%% output:
%{
fs_indices - a 1xK matrix containing the indices of features determined to 
be useful by sequential forward selection. K is the number of features
sequential forward selection chose before deciding that adding additional
features would not be useful.
%}

%% Classification code
%clear feature_index final_index score_new score_old f train_acc score_track count data_aug  MdlLinear train_pred train_ConfMat max_index 
%filter_select_count = 10;

%Getting the number of features
feature_index = (1:filter_select_count);%This vector is all indexes from 1 to len(filter_select_count)
final_index = zeros(1,filter_select_count); %This vector is our final index

%defining score for evaluation criterion
score_new = 0.00001;
score_old = 0;
f = 0;%keeps count of relevant features from the algorithm below

%loop goes on as long as the current combination has a better train acc.
%score than the previous one.
while (score_old<score_new && f<filter_select_count)
    f = f+1;
    score_track = zeros(1,filter_select_count);
    for i =1:1:length(feature_index)
        %Current index of final feature vector is dynamic and upto
        %evaluation
        final_index(f) = feature_index(i);
        %creating a data array that only has training data corresponding to features
        %from final index vector
        for count = 1:1:f
            data_aug(:,count) = train_data(:,final_index(count)); %#ok<*SAGROW>
        end
        %training model for accuracy scores
        MdlLinear = fitcdiscr(data_aug,train_labels);
        %MdlLinear = fitctree(data_aug,train_labels);

        %   Find the training accurracy 
        train_pred = predict(MdlLinear,data_aug);

        %   Create confusion matrix
        train_ConfMat = confusionmat(train_labels,train_pred);
        train_ConfMat = train_ConfMat./(meshgrid(countcats(categorical(train_labels)))');
        %   mean group accuracy and std
        train_acc = mean(diag(train_ConfMat));
        score_track(i) = train_acc;
    end
    score_old = score_new; %updating old score
    %updating new score and getting index of max score
    [score_new,max_index] = max(score_track); 
    %updating current index of final index vector
    final_index(f) = feature_index(max_index);
    %removing that index from feature index file
    feature_index(max_index) = [];
end

%% Implementation
fs_indices = final_index(1:f);
% TODO: REPLACE DUMMY IMPLEMENTATION