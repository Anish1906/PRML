% Start code for Project 2: Feature Selection
% CSE583/EE552 PRML
% TA: Shimian Zhang, Spring 2022
% TA: Addison Petro, Spring 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Plots a histogram of features with corresponding values
%}
function plotFeat(FeatStat,FeatNames,num_on_bar,use_title,xlabel_in,filename)
% FeatStat: nx2 where n is the number of features.  
%     The second dimension is feature number and then score
% FeatNames: Names of all the features
% num_on_bar the number of features to show on the bar graph


%% Sort top feature and plot on bar graph
if (size(FeatStat, 1) < num_on_bar)
    num_on_bar = size(FeatStat, 1);
end
fig = figure(6);
set(fig, 'visible', 'off');
FeatStat = sortrows(FeatStat,-2);
barh(FeatStat(num_on_bar:-1:1,2));
FeatNames = FeatNames(FeatStat(:,1));
set(fig.CurrentAxes,'YTick', 1:num_on_bar,'YTickLabel',FeatNames(num_on_bar:-1:1),'FontSize', 14);
set(fig, 'Position', [100, 100, 600, 200+20*num_on_bar]);
ylim([.5,num_on_bar+.5]);
grid on
xlabel(xlabel_in,'FontSize', 18);
ylabel('Features','FontSize', 18);
title(use_title);
export_fig(fig, filename, '-png', '-transparent');