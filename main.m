%% Note

% You do *not* have to modify this script.

%% Cleanup, turn the warnings off and add the current folder to the path

close all;
clearvars;
clc;

warning('off');

addpath(genpath(pwd));

%% Load the data

n_test     = 1000; % number of test observations (n_test = Inf returns the entire test set of 10000 observations)
n_training = 4000; % number of training observations (n_training = Inf returns the entire training set  of 60000 observations)

[T_test, T_training, X_test, X_training] = load_data(n_test, n_training);

%% Initialize and vectorize the weights and biases

args.filterDim  = 9;  % Filter size for conv layer
args.imageDim   = 28; % Image size for input layer
args.numClasses = 10; % Number of classes (MNIST images fall into 10 classes)
args.numFilters = 20; % Number of filters for conv layer
args.poolDim    = 2;  % Pooling dimension, (should divide imageDim-filterDim+1)

theta = cnnInitParams(args);

[W_1, W_2, b_1, b_2] = cnnParamsToStack(theta, args);

%% Train the convolutional neural network

options.alpha     = 1e-1;  % learning rate
options.epochs    = 1;     % number of passes over the training set
options.minibatch = 32;    % number of observations in a mini-batch
options.momentum  = 95e-2; % momentum

opttheta = minFuncSGD(@(x, y, z) cross_entropy(args, x, y, z), theta, X_training, T_training, options);

%% Unvectorize the weights and biases

[W_1, W_2, b_1, b_2] = cnnParamsToStack(opttheta, args);

%% Compute the predictions

[~, T_test_hat] = forwardprop(args, b_1, b_2, W_1, W_2, X_test);

%% Compute the test accuracy

[~, I]     = max(T_test);
[~, I_hat] = max(T_test_hat);

accuracy = 100 * mean(I == I_hat);

disp(['Test accuracy = ' num2str(accuracy) '%']);

%% Visualize the internal representations of the hidden units

W = padarray(W_1, [1 1 0]);
d = size(W);
W = reshape(W, [], d(3));
W = col2im(W, d([1 2]), [4 5] .* d([1 2]), 'distinct'); % if args.numFilters ~= 20 then change [4 5] to [a b] such that args.numFilters == a * b 

imagesc(W);

colormap(gray);

axis off;

