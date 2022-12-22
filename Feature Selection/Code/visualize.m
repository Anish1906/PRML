% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: Visualization script
%}

%% Clean up the workspace
clear all;
close all;
clc;
addpath export_fig;


%% Setup runtime variables
% File information
dataset = 'Taiji_subset';
out_dir = 'output/fw';

% Set number of features to be visualized
num_visualized = 30;


%% Load data
load(dataset);
fprintf("Data loaded\n");
num_features = size(Taiji_data, 2);
num_forms = size(form_names, 1);


%% Perform Visualizations
fprintf("Starting visualization\n");

% Setup colormap for confusion matrices
load('colormaps.mat');
conf_mat_colors = flipud(cetcbl2);

% Load overall classification data
overall_filename = [out_dir, '/overall.mat'];
load(overall_filename);

% Render overall subject training and testing rates
fig_subj_rates = figure(3);
set(fig_subj_rates, 'visible', 'off');
bar_subj_rates = bar([subj_rate_train, subj_rate_test]* 100);
legend('Location', 'northoutside', 'Orientation', 'horizontal');
title(['Overall Subject Rates'], 'Interpreter', 'none');
xlabel('Subject Number');
ylabel('Classification rate (%)');
ylim([0, 100]);
set(fig_subj_rates, 'Position', [100, 100, 800, 600]);
ax = gca;
ax.XAxis.TickLength = [0, 0];
ax.YTick = 0:5:100;
ax.YGrid = 'on';
train_mean = mean(subj_rate_train * 100);
train_std = std(subj_rate_train * 100);
test_mean = mean(subj_rate_test * 100);
test_std = std(subj_rate_test * 100);
line(xlim, [train_mean, train_mean], 'Color', '#0072BD', 'LineWidth', 2);
line(xlim, [test_mean, test_mean], 'Color', '#D95319', 'LineWidth', 2);
line(xlim, [train_mean + train_std, train_mean + train_std], 'Color', '#0072BD', 'LineWidth', 2, 'LineStyle','--');
line(xlim, [test_mean + test_std, test_mean + test_std], 'Color', '#D95319', 'LineWidth', 2, 'LineStyle', '--');
line(xlim, [train_mean - train_std, train_mean - train_std], 'Color', '#0072BD', 'LineWidth', 2, 'LineStyle','--');
line(xlim, [test_mean - test_std, test_mean - test_std], 'Color', '#D95319', 'LineWidth', 2, 'LineStyle', '--');
h = flip(get(ax, 'Children'));
hmod = [h(1), h(3), h(5), h(2), h(4), h(6)];
legend_str = {'Training', 'Training Mean', '±1 σ Training', 'Testing', 'Testing Mean', '±1 σ Testing'};
legend(hmod, legend_str, 'Location', 'northoutside', 'Orientation', 'horizontal', 'NumColumns', 3, 'FontSize', 14);
subj_rates_filename = [out_dir, '/subject_rates'];
export_fig(fig_subj_rates, subj_rates_filename, '-png', '-transparent');

% Render overall per class training data
fig_class_rates_train = figure(4);
set(fig_class_rates_train, 'visible', 'off');
bar_class_rates_train = bar(0:1:num_forms, overall_per_class_train * 100);
title(['Overall Training Rates by Class'], 'Interpreter', 'none');
xlabel('Class Number');
ylabel('Classification rate (%)');
ylim([0, 100]);
set(fig_class_rates_train, 'Position', [100, 100, 800, 600]);
ax = gca;
ax.XAxis.TickLength = [0, 0];
ax.YTick = 0:5:100;
ax.XTick = 0:1:num_forms;
ax.YGrid = 'on';
per_class_train_mean = mean(overall_per_class_train * 100);
per_class_train_std = std(overall_per_class_train * 100);
line(xlim, [per_class_train_mean, per_class_train_mean], 'Color', '#000000', 'LineWidth', 2);
line(xlim, [per_class_train_mean + per_class_train_std, per_class_train_mean + per_class_train_std], 'Color', '#000000', 'LineWidth', 2, 'LineStyle','--');
line(xlim, [per_class_train_mean - per_class_train_std, per_class_train_mean - per_class_train_std], 'Color', '#000000', 'LineWidth', 2, 'LineStyle','--');
h = flip(get(ax, 'Children'));
legend_str = {'Class Rates', 'Mean', '±1 Std. Dev.'};
legend(h(1:3), legend_str, 'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 14);
class_rates_filename_train = [out_dir, '/class_rates_train'];
export_fig(fig_class_rates_train, class_rates_filename_train, '-png', '-transparent');

% Render overall per class testing data
fig_class_rates_test = figure(5);
set(fig_class_rates_test, 'visible', 'off');
bar_class_rates_test = bar(0:1:num_forms, overall_per_class_test * 100);
bar_class_rates_test.FaceColor = '#D95319';
title(['Overall Testing Rates by Class'], 'Interpreter', 'none');
xlabel('Class Number');
ylabel('Classification rate (%)');
ylim([0, 100]);
set(fig_class_rates_test, 'Position', [100, 100, 800, 600]);
ax = gca;
ax.XAxis.TickLength = [0, 0];
ax.YTick = 0:5:100;
ax.XTick = 0:1:num_forms;
ax.YGrid = 'on';
per_class_test_mean = mean(overall_per_class_test * 100);
per_class_test_std = std(overall_per_class_test * 100);
line(xlim, [per_class_test_mean, per_class_test_mean], 'Color', '#000000', 'LineWidth', 2);
line(xlim, [per_class_test_mean + per_class_test_std, per_class_test_mean + per_class_test_std], 'Color', '#000000', 'LineWidth', 2, 'LineStyle','--');
line(xlim, [per_class_test_mean - per_class_test_std, per_class_test_mean - per_class_test_std], 'Color', '#000000', 'LineWidth', 2, 'LineStyle','--');
h = flip(get(ax, 'Children'));
legend_str = {'Class Rates', 'Mean', '±1 Std. Dev.'};
legend(h(1:3), legend_str, 'Location', 'northoutside', 'Orientation', 'horizontal', 'FontSize', 14);
class_rates_filename_test = [out_dir, '/class_rates_test'];
export_fig(fig_class_rates_test, class_rates_filename_test, '-png', '-transparent');

% Render overall training confusion matrix
fig_conf_train = figure(1);
set(fig_conf_train, 'visible', 'off');
hmap_conf_train = heatmap(overall_train_mat, 'Colormap', conf_mat_colors);
title(['Overall Training']);
xlabel('True class');
ylabel('Predicted class');
caxis(hmap_conf_train, [0, 1]);
set(fig_conf_train, 'Position', [100, 100, 1000, 1000]);
ax = gca;
ax.XData = 0:num_forms;
ax.YData = 0:num_forms;
mat_filename_train = [out_dir, '/train_mat_overall'];
export_fig(fig_conf_train, mat_filename_train, '-png', '-transparent');

% Render ovearll testing confusion matrix
fig_conf_test = figure(2);
set(fig_conf_test, 'visible', 'off');
hmap_conf_test = heatmap(overall_test_mat, 'Colormap', conf_mat_colors);
title(['Overall Testing']);
xlabel('True class');
ylabel('Predicted class');
caxis(hmap_conf_test, [0, 1]);
set(fig_conf_test, 'Position', [100, 100, 1000, 1000]);
ax = gca;
ax.XData = 0:num_forms;
ax.YData = 0:num_forms;
mat_filename_test = [out_dir, '/test_mat_overall'];
export_fig(fig_conf_test, mat_filename_test, '-png', '-transparent');

% Render most commonly selected features
selected_counts = zeros(num_features, 1);
for subj_num = 1:num_subjects
    subject_filename = [out_dir, '/subject_', num2str(subj_num), '.mat'];
    load(subject_filename);
    for val = 1:filter_select_count
        selected_counts(filter_indices(val)) = selected_counts(filter_indices(val)) + 1;
    end
end
filter_plot_info = zeros(num_features, 2);
filter_plot_info(:, 1) = [1:num_features]';
filter_plot_info(:, 2) = selected_counts;
filter_plot_filename = [out_dir, '/selected_features'];
plotFeat(filter_plot_info, feature_names, num_visualized, 'Most Frequently Selected Features', 'Times Selected', filter_plot_filename);

% Render average filter scores
average_filter_scores = zeros(num_features, 1);
for subj_num = 1:num_subjects
    subject_filename = [out_dir, '/subject_', num2str(subj_num), '.mat'];
    load(subject_filename);
    average_filter_scores = average_filter_scores + filter_scores';
end
average_filter_scores = average_filter_scores / num_subjects;
average_filter_plot_info = zeros(num_features, 2);
average_filter_plot_info(:, 1) = [1:num_features]';
average_filter_plot_info(:, 2) = average_filter_scores;
average_filter_plot_filename = [out_dir, '/average_filters'];
plotFeat(average_filter_plot_info, feature_names, num_visualized, 'Average Filter Scores', 'filter', average_filter_plot_filename);

fprintf("Visualization complete\n");
