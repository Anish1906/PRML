% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: Main driver script
%}

%% Clean up the workspace
clear all;
close all;
clc;


%% Setup runtime variables
% File information
dataset = 'Taiji_subset';
out_dir = 'output';

% Set number to be selected by filter
% You will need to change this
filter_select_count = 100; % TODO: CHANGE VALUE


%% Create output directory if it doesn't exist yet
if (~exist(out_dir, 'dir'))
    mkdir(out_dir);
end


%% Load data
load(dataset);
fprintf("Data loaded\n");
num_features = size(Taiji_data, 2);
num_forms = size(form_names, 1);


%% Perform Training and Evaluation
% Set up empty classification rate variables
subj_rate_train = zeros(num_subjects, 1);
subj_per_class_train = zeros(num_subjects, num_forms + 1);
subj_rate_test = zeros(num_subjects, 1);
subj_per_class_test = zeros(num_subjects, num_forms + 1);
overall_train_mat = zeros(num_forms + 1);
overall_test_mat = zeros(num_forms + 1);

for subj_num = 1:num_subjects
    fprintf("Starting training and evaluation for subject %d\n", subj_num);
    subj_start = tic;

    % Split the data into training and testing
    [train_data, train_labels, test_data, test_labels]= split(Taiji_data, labels, sub_info, subj_num);

    % =====================================================
    % Normalize the data here (see normalize_data function)
    % =====================================================
    % You need to normalize each feature such that it exists on the
    % range [0, 1] so that no features will have unfairly weighted
    % means or variances for the filter method
    [train_data, norm_min, norm_diff] = normalize_data(train_data);
    test_data = normalize_data(test_data, norm_min, norm_diff);

    % =====================================================
    % Perform feature filtering here
    % =====================================================
    if (num_features < filter_select_count)
        filter_select_count = num_features;
    end
    % Get the best features based on filtering method
    % TODO: IMPLEMENT filterMethod()
    [filter_indices, filter_scores] = filterMethod(train_data, train_labels);
    % Reduce the data to the top features
    train_data = train_data(:, filter_indices(1:filter_select_count));
    test_data = test_data(:, filter_indices(1:filter_select_count));

    % =====================================================
    % Perform sequential forward selection here
    % =====================================================
    % TODO: IMPLEMENT forwardSelection()
    fs_indices = forwardSelection(train_data, train_labels, filter_select_count);
    % Reduce data to the top features
    train_data = train_data(:, fs_indices);
    test_data = test_data(:, fs_indices);

    % =====================================================
    % Train a model here
    % =====================================================
    % Feel free to use classifiers other than LDA
    % TODO: CHANGE CLASSIFIER
    % Note: This TODO is not required but highly recommended
    model = fitcdiscr(train_data, train_labels, 'discrimType', 'pseudoLinear');
    %model = fitcknn(train_data, train_labels, 'NumNeighbors',3);
    %model = fitctree(train_data, train_labels);
    
    % =====================================================
    % Evaluate the model here
    % =====================================================
    % Get predictions
    subj_pred_train = predict(model, train_data);
    subj_pred_test = predict(model, test_data);

    % Compute subject overall accuracy
    subj_acc_train = mean(subj_pred_train == train_labels);
    subj_rate_train(subj_num) = subj_acc_train;
    subj_acc_test = mean(subj_pred_test == test_labels);
    subj_rate_test(subj_num) = subj_acc_test;

    % Compute subject training per-class accuracy
    subj_class_totals_train = zeros(1, num_forms + 1);
    subj_class_correct_train = zeros(1, num_forms + 1);
    conf_mat_train = zeros(num_forms + 1);
    for elem = 1:length(subj_pred_train)
        pred_val = double(string(subj_pred_train(elem)));
        real_val = double(string(train_labels(elem)));
        subj_class_totals_train(real_val + 1) = subj_class_totals_train(real_val + 1) + 1;
        if (pred_val == real_val)
            subj_class_correct_train(real_val + 1) = subj_class_correct_train(real_val + 1) + 1;
        end
        conf_mat_train(pred_val + 1, real_val + 1) = conf_mat_train(pred_val + 1, real_val + 1) + 1;
    end
    subj_classes_train = subj_class_correct_train ./ subj_class_totals_train;
    subj_per_class_train(subj_num, :) = subj_classes_train;

    % Compute subject testing per-class accuracy
    subj_class_totals_test = zeros(1, num_forms + 1);
    subj_class_correct_test = zeros(1, num_forms + 1);
    conf_mat_test = zeros(num_forms + 1);
    for elem = 1:length(subj_pred_test)
        pred_val = double(string(subj_pred_test(elem)));
        real_val = double(string(test_labels(elem)));
        subj_class_totals_test(real_val + 1) = subj_class_totals_test(real_val + 1) + 1;
        if (pred_val == real_val)
            subj_class_correct_test(real_val + 1) = subj_class_correct_test(real_val + 1) + 1;
        end
        conf_mat_test(pred_val + 1, real_val + 1) = conf_mat_test(pred_val + 1, real_val + 1) + 1;
    end
    subj_classes_test = subj_class_correct_test ./ subj_class_totals_test;
    subj_per_class_test(subj_num, :) = subj_classes_test;

    % Normalize columns of confusion matrices
    for col = 1:size(conf_mat_train, 1)
        conf_mat_train(col, :) = conf_mat_train(col, :) ./ subj_class_totals_train;
        conf_mat_test(col, :) = conf_mat_test(col, :) ./ subj_class_totals_test;
    end

    % Replace all NaNs with 0
    subj_classes_train(isnan(subj_classes_train)) = 0.0;
    subj_classes_test(isnan(subj_classes_test)) = 0.0;
    conf_mat_train(isnan(conf_mat_train)) = 0.0;
    conf_mat_test(isnan(conf_mat_test)) = 0.0;

    % Overall confusion matrix addition
    overall_train_mat = overall_train_mat + (1 / num_subjects) * conf_mat_train;
    overall_test_mat = overall_test_mat + (1 / num_subjects) * conf_mat_test;

    % Display results
    fprintf("Subject %d training accuracy: %.2f%%\n", subj_num, subj_acc_train * 100);
    fprintf("Subject %d testing accuracy: %.2f%%\n", subj_num, subj_acc_test * 100);

    % =====================================================
    % Save everything in case you need it later
    % =====================================================
    subject_filename = [out_dir, '/subject_', num2str(subj_num), '.mat'];
    save(subject_filename, 'model', 'filter_select_count', 'filter_indices', 'filter_scores', 'fs_indices', ...
                           'subj_acc_train', 'subj_classes_train', 'conf_mat_train', ...
                           'subj_acc_test', 'subj_classes_test', 'conf_mat_test');
    fprintf("Subject %d training and evaluation completed in %.3f seconds\n", subj_num, toc(subj_start));
    fprintf("================================================================================\n");
end
% Replace overall NaNs with 0
subj_per_class_train(isnan(subj_per_class_train)) = 0.0;
subj_per_class_test(isnan(subj_per_class_test)) = 0.0;

% Compute overall results
overall_rate_train = mean(subj_rate_train);
overall_per_class_train = mean(subj_per_class_train);
overall_rate_test = mean(subj_rate_test);
overall_per_class_test = mean(subj_per_class_test);

% Display overall results
fprintf("Overall training accuracy: %.2f%%\n", overall_rate_train * 100);
fprintf("Overall testing accuracy: %.2f%%\n", overall_rate_test * 100);

% Save overall results
overall_filename = [out_dir, '/overall.mat'];
save(overall_filename, 'overall_rate_train', 'overall_per_class_train', 'subj_rate_train', 'subj_per_class_train', 'overall_train_mat', ...
                       'overall_rate_test', 'overall_per_class_test', 'subj_rate_test', 'subj_per_class_test', 'overall_test_mat');          
fprintf("Training and evaluation complete\n");
fprintf("================================================================================\n");
