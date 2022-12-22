# PRML Project 3, Spring 2022

Following is the readme for the Deep Learning project folder

## Codes included

- **start.m**: the main driver script for this project
  - Loads the data for training and testing
  - Has two main sections, Transfer Network and Self-Designed Network.
  - AlexNet has one training block with 5 epochs
  - Self Designed network has 3 blocks, 5+3+3 epochs.
  - Visualisation for each is given below respective network methods.
  
- **dataAugmentation.m**: the data augmentation script to generate your own dataset
  - Loads the initial dataset
  - Has two blocks, one creates augmented dataset for training, other for testing.
  - Training has 5 loopings for augmentation, Testing has one looping
  - Plots histogram of parameters right after augmentation

