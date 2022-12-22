% Data Augmentation for project 3: Deep Learning
% PRML, CSE583/EE552
% TA: Shimian Zhang, Feb 2022
% TA: Addison Petro, Feb 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: This script takes in the training and testing wallpaper dataset and augments it to create a bigger more expansive dataset.
%}
%% Clean up the workspace
clear all;
close all;
clc;

%% Clean up the workspace
clear all;
close all;
clc;

%% Load original dataset
dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train';
test_folder  = 'test';

aug_train_folder = 'train_aug';
aug_test_folder  = 'test_aug';

if exist(fullfile(dataDir,aug_train_folder), 'dir') == 0
    mkdir(fullfile(dataDir,aug_train_folder));
    for cls = Symmetry_Groups
        mkdir(fullfile(dataDir,aug_train_folder, cls{1}));
    end
end

if exist(fullfile(dataDir,aug_test_folder), 'dir') == 0
    mkdir(fullfile(dataDir,aug_test_folder));
    for cls = Symmetry_Groups
        mkdir(fullfile(dataDir,aug_test_folder, cls{1}));
    end
end

fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);

fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%% Data augmentation for train set
% =====================================================
% You need to do data augmentation, with at least 5 times per training
% image

% TODO: IMPLEMENT DATA AUGMENTATION HERE
% =====================================================
% augmenter = imageDataAugmenter( ...
%     'RandRotation',[0 360],...
%     'RandScale',[1 2],...
%     'RandXTranslation', [-10 10],...
%     'RandYTranslation', [-10 10]);
rng(0,'twister'); %random number generator
%We define our parameters for the random parameter generation
rot_a = 0;
rot_b = 360;
scale_a = 1;
scale_b = 2;
xtrans_a = -10;
xtrans_b = 10;
ytrans_a = -10;
ytrans_b = 10;
index = 1

%generating augmented images for training
for i = 1:length(Symmetry_Groups)
    fold = dir(fullfile(dataDir,train_folder,Symmetry_Groups{i}, '*.png'));
    for j = 1:5
        for k = 1:1000
            I = imread(fullfile(dataDir,train_folder,Symmetry_Groups{i},fold(k).name));
            rot_rand = (rot_b-rot_a).*rand() + rot_a;
            scale_rand = (scale_b-scale_a).*rand() + scale_a;
            xtrans_rand = floor((xtrans_b-xtrans_a).*rand() + xtrans_a);
            ytrans_rand = floor((ytrans_b-ytrans_a).*rand() + ytrans_a);
            I_rot = imrotate(I,rot_rand);%rotation
            I_scale = imresize(I_rot,scale_rand);%scaling
            I_trans = imtranslate(I_scale,[xtrans_rand, ytrans_rand],'FillValues',255);%translation
            window = centerCropWindow2d(size(I_trans),[128 128]);
            I_crop = imcrop(I_trans,window);%crop
            I_final = imresize(I_crop,2); %resize to [256 256]
            rot(index) = rot_rand; %#ok<SAGROW>
            scale(index) = scale_rand; %#ok<SAGROW>
            xtrans(index) = xtrans_rand; %#ok<SAGROW>
            ytrans(index) = ytrans_rand; %#ok<SAGROW>
            imwrite(I_final,(fullfile(dataDir,aug_train_folder,Symmetry_Groups{i},strcat(Symmetry_Groups{i},'_',int2str(index), '.png'))));
            index = index + 1
        end
    end
end
% =====================================================
% You should plot a histogram of the scales, rotations, and translations 
% to show the distribution of the augmentations. 

% TODO: PLOT THE DISTRIBUTION OF AUGMENTATION VARIABLES 
% =====================================================
figure,histogram(rot,'FaceColor','r')
grid on;
xlabel('Angle of Rotation', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Rotations(Training)', 'FontSize', 14);
figure,histogram(scale,'FaceColor','g')
grid on;
xlabel('Scaling value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Uniform Scaling(Training)', 'FontSize', 14);
figure,histogram(xtrans,'FaceColor','b')
grid on;
xlabel('Translation in x Value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Translation in X-direction(Training)', 'FontSize', 14);
figure,histogram(ytrans,'FaceColor','m')
grid on;
xlabel('Translation in y Value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Translation in Y-direction(Training)', 'FontSize', 14);

%% Data augmentation for test set
% =====================================================
% You need to do data augmentation, with at least once per testing
% image

% TODO: IMPLEMENT DATA AUGMENTATION HERE
% =====================================================
%Data augmentation for testing
count = 1
for i = 1:length(Symmetry_Groups)
    fold = dir(fullfile(dataDir,test_folder,Symmetry_Groups{i}, '*.png'));
    for j = 1:1
        for k = 1:1000
            I = imread(fullfile(dataDir,train_folder,Symmetry_Groups{i},fold(k).name));
            rot_rand = (rot_b-rot_a).*rand() + rot_a;
            scale_rand = (scale_b-scale_a).*rand() + scale_a;
            xtrans_rand = floor((xtrans_b-xtrans_a).*rand() + xtrans_a);
            ytrans_rand = floor((ytrans_b-ytrans_a).*rand() + ytrans_a);
            I_rot = imrotate(I,rot_rand);%rotation
            I_scale = imresize(I_rot,scale_rand);%scaling
            I_trans = imtranslate(I_scale,[xtrans_rand, ytrans_rand],'FillValues',255);%translation
            window = centerCropWindow2d(size(I_trans),[128 128]);
            I_crop = imcrop(I_trans,window);%cropping
            I_final = imresize(I_crop,2); %resize to [256 256]
            rot_test(count) = rot_rand; %#ok<SAGROW>
            scale_test(count) = scale_rand; %#ok<SAGROW>
            xtrans_test(count) = xtrans_rand; %#ok<SAGROW>
            ytrans_test(count) = ytrans_rand; %#ok<SAGROW>
            imwrite(I_final,(fullfile(dataDir,aug_test_folder,Symmetry_Groups{i},strcat(Symmetry_Groups{i},'_',int2str(count), '.png'))));
            count = count + 1
        end
    end
end
% =====================================================
% You should plot a histogram of the scales, rotations, and translations 
% to show the distribution of the augmentations. 

% TODO: PLOT THE DISTRIBUTION OF AUGMENTATION VARIABLES 
% =====================================================
figure,histogram(rot_test,'FaceColor','r')
grid on;
xlabel('Angle of Rotation', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Rotations(Testing)', 'FontSize', 14);
figure,histogram(scale_test,'FaceColor','g')
grid on;
xlabel('Scaling value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Uniform Scaling(Testing)', 'FontSize', 14);
figure,histogram(xtrans_test,'FaceColor','b')
grid on;
xlabel('Translation in x Value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Translation in X-direction(Testing)', 'FontSize', 14);
figure,histogram(ytrans_test,'FaceColor','m')
grid on;
xlabel('Translation in y Value', 'FontSize', 14);
ylabel('Number of times used', 'FontSize', 14);
title('Histogram of Translation in Y-direction(Testing)', 'FontSize', 14);