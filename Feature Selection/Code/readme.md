#Anish Phule, PRML Project 2, Spring 2022

The following is the readme file for the Feature Selection project code.

##start_classification.m
This code is the main code that 
-takes in the data values
-splits them
-applies filtering and wrapping
-performs Lose One Subject Out (LOSO)
-gives training and testing accuracy results for all subjects

##filterMethod.m
This code performs filtering and gives us top N filtered features.
-Takes in data and labels
-calculates Variance ratio for features
-arranges features according to their variance ratio
-gives us top features

##forwardSelection.m
This code performs wrapping.
-takes in the sorted labels and data after filtering
-performs training accuracy scoring on each feature 
-gives optimised feature list

##split.m
splits the dataset into training and testing data based on subject number


##visualize.m
a script to visualize the results in your output folder