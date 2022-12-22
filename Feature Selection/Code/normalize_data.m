% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Normalizes input data to be on range 0-1
%}
function [norm_data, min_vals, diff_vals] = normalize_data(varargin)
%% Get the maximum and minimum value from each column
% For normalizing training data, only pass in train_data
if (nargin == 1)
    in_data = varargin{1};
    max_vals = max(in_data);
    min_vals = min(in_data);
    diff_vals = max_vals - min_vals;
% For normalizing test data, pass in test_data, min_vals, diff_vals
% min_vals and diff_vals will be returned from normalizing training data
elseif (nargin == 3)
    in_data = varargin{1};
    min_vals = varargin{2};
    diff_vals = varargin{3};
end

%% Normalize the data
% Normalized value = (orignal value - min)/(max - min)
norm_data = zeros(size(in_data));
for i = 1:size(in_data, 1)
    norm_data(i,:) = (in_data(i,:) - min_vals)./(diff_vals);
end
% Replace all NaN with zeros
norm_data(isnan(norm_data)) = 0.0;
norm_data(isinf(norm_data)) = 0.0;