% Starter code for project 3: Deep Learning
% PRML, CSE583/EE552
% TA: Shimian Zhang, Feb 2022
% TA: Addison Petro, Feb 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name: Anish Phule
    PSU Email ID: asp5607@psu.edu
    Description: This script performs the Deep Learning program.
%}
%% Clean up the workspace
clear all;
close all;
clc;

%% Load dataset
dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1); % For reproducibility

Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

%train_folder = 'train';
%test_folder  = 'test';

% =====================================================
% Uncomment after you create the augmentation dataset
% =====================================================
train_folder = 'train_aug';
test_folder  = 'test_aug';

fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%% Transfer Learning : AlexNet

net = alexnet;
layersTransfer = net.Layers(1:end-3);
numClasses = 17;
net = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer
    ];

augimdstrain = augmentedImageDatastore([227 227 3],train,'colorpreprocessing','gray2rgb');
augimdsval = augmentedImageDatastore([227 227 3],val,'colorpreprocessing','gray2rgb');
augimdstest = augmentedImageDatastore([227 227 3],test,'colorpreprocessing','gray2rgb');

options = trainingOptions('adam','MaxEpochs',5,... 
    'InitialLearnRate',1e-5,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', 200, ...
    'Plots','training-progress');

t = tic;
netTransfer = trainNetwork(augimdstrain,net,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

YTest = classify(netTransfer,augimdsval);
val_acc = mean(YTest==augimdsval.Labels)

disp('Evaluation...')

YTrain = classify(netTransfer,augimdstrain);
train_acc = mean(YTrain==augimdstrain.Labels)

YVal = classify(netTransfer,augimdsval);
val_acc = mean(YVal==augimdsval.Labels)

YTest = classify(netTransfer,augimdstest);
test_acc = mean(YTest==augimdstest.Labels)

% =====================================================
% Plot confusion matrix
% =====================================================
figure()
confusionchart(augimdstrain.Labels, YTrain);

figure()
confusionchart(augimdsval.Labels, YVal);

figure()
confusionchart(augimdstest.Labels, YTest);

disp('Visualization...')

% =====================================================
% Show the details of the network
% =====================================================
analyzeNetwork(net);

% =====================================================
% Visualize the first conv layer filters

% NOTE:You may need to modify the layer name based on your customized
% network. The layer name can be found by analyzeNetwork().
% =====================================================
layer = 'conv';
c_num = 20;
channels = 1:c_num;
I = deepDreamImage(netTransfer,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);

figure()
for i = 1:c_num
    subplot(4,5,i)
    imshow(I(:,:,:,i))
end

% Do a t-SNE multidimensional reduction on the last fc layer activations
% TODO: IMPLEMENT T-SNE
% =====================================================
t1_fc = 'fc';
t1_act = activations(net,...
    augimdsval,t1_fc, "OutputAs", "rows");

t1_tsne = tsne(t1_act);

figure;
gscatter(t1_tsne(:,1), t1_tsne(:,2), val.Labels);

%% Define the network and training parameters(Self-designed)
rng('default');
% nTraining = length(train.Labels);

% =====================================================
% Define the Network Structure here, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 

% TODO: DESIGN YOUR OWN NETWORK ON AUGMENTED DATASET
% =====================================================

layers = [
    imageInputLayer([256 256 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,20,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    batchNormalizationLayer %Normalization layer
    reluLayer();  % ReLU layer
    convolution2dLayer(5,15,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 15, 5x5 filters
    batchNormalizationLayer %Normalization layer
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(5,10,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 10, 5x5 filters
    batchNormalizationLayer %Normalization layer
    reluLayer();  % ReLU layer
    convolution2dLayer(3,20,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 3x3 filters
    batchNormalizationLayer %Normalization layer
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(3,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 40, 3x3 filters
    reluLayer();  % ReLU layer
    batchNormalizationLayer %Normalization layer
    convolution2dLayer(3,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 40, 3x3 filters
    batchNormalizationLayer %Normalization layer
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(50,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20); % Fullly connected layer with 50 activations
    reluLayer();  % ReLU layer
    fullyConnectedLayer(50,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20); % Fullly connected layer with 50 activations
    reluLayer();  % ReLU layer
    dropoutLayer(.4); % Dropout layer
    fullyConnectedLayer(17,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% =====================================================
% Set the training options here

% TODO: SETUP SUITABLE TRAINING PARAMETERS
% =====================================================
numEpochs = 5;
options = trainingOptions('sgdm','MaxEpochs',5,... 
    'InitialLearnRate',1e-3,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', 120, ...
    'Plots','training-progress');

%% 1-st training phase
% =====================================================
% Train the network, info contains information about the training accuracy
% and loss
% =====================================================


 t = tic;
[net,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

% =====================================================
% Test on the validation data
% =====================================================
YTest = classify(net,val);
val_acc = mean(YTest==val.Labels)



%% 2-nf training phase (Optional)
% =====================================================
% Here is an example of continued training.
% It seems like it isn't converging after looking at the graph but lets
% try dropping the learning rate to show you how.  
% =====================================================

% =====================================================
% Use the following function to load from a specified checkpoint
% load('modelCheckpoints/*.mat','net')
% =====================================================

options = trainingOptions('sgdm','MaxEpochs',3,...
    'InitialLearnRate',1e-3,... % learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', 120, ...
    'Plots','training-progress');

 t = tic;
[net,info2] = trainNetwork(train,net.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));
% =====================================================
% Test on the validation data
% =====================================================
YTest = classify(net,val);
val_acc = mean(YTest==val.Labels)

%% 3-nf training phase (Optional)
% =====================================================
% Here is an example of continued training.
% It seems like it isn't converging after looking at the graph but lets
% try dropping the learning rate to show you how.  
% =====================================================

% =====================================================
% Use the following function to load from a specified checkpoint
% load('modelCheckpoints/*.mat','net')
% =====================================================

options = trainingOptions('sgdm','MaxEpochs',3,...
    'InitialLearnRate',1e-4,... % learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', 120, ...
    'Plots','training-progress');

 t = tic;
[net,info2] = trainNetwork(train,net.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));
% =====================================================
% Test on the validation data
% =====================================================
YTest = classify(net,val);
val_acc = mean(YTest==val.Labels)


%% Performance evaluation
disp('Evaluation...')

YTrain = classify(net,train);
train_acc = mean(YTrain==train.Labels)

YVal = classify(net,val);
val_acc = mean(YVal==val.Labels)

YTest = classify(net,test);
test_acc = mean(YTest==test.Labels)

% =====================================================
% Plot confusion matrix
% =====================================================
figure()
confusionchart(train.Labels, YTrain);

figure()
confusionchart(val.Labels, YVal);

figure()
confusionchart(test.Labels, YTest);

%% Model Visualization
disp('Visualization...')

% =====================================================
% Show the details of the network
% =====================================================
analyzeNetwork(net);

% =====================================================
% Visualize the first conv layer filters

% NOTE:You may need to modify the layer name based on your customized
% network. The layer name can be found by analyzeNetwork().
% =====================================================
layer = 'conv';
c_num = 20;
channels = 1:c_num;
I = deepDreamImage(net,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);

figure()
for i = 1:c_num
    subplot(4,5,i)
    imshow(I(:,:,:,i))
end
% =====================================================
% Do a t-SNE multidimensional reduction on the last fc layer activations
% TODO: IMPLEMENT T-SNE
% =====================================================
t1_fc = 'fc_3';
t1_act = activations(net,...
    val,t1_fc, "OutputAs", "rows");

t1_tsne = tsne(t1_act);

figure;
gscatter(t1_tsne(:,1), t1_tsne(:,2), val.Labels);

%% Another example of network design
% =====================================================
% Here is an example of another network design.
% Here we add another set of "CONV -> ReLU -> POOL ->" to make the network:
% CONV -> ReLU -> POOL -> CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
% =====================================================
layers = [
    imageInputLayer([256 256 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,20,'Padding',[2 2],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(3,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(25); % Fullly connected layer with 25 activations
    dropoutLayer(.25); % Dropout layer
    fullyConnectedLayer(17); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

