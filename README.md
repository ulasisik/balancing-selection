# balancing-selection

This repo contains the code used for distinguishing between balancing selection and incomplete sweep using deep learning.

### Simulations
We performed simulations for a neutral(NE) and three different selection scenarios, including incomplete sweep(IS), overdominance(OD) and negative frequency dependent selection(FD).
To run simulations, user should run doXX(NE/IS/OD/FD).sh scripts located in `simulation` folder. 
Different selection start times, dominance and selection coefficients can be set by modifying doXX.sh scripts. 
In order to change parameters of demographic model, mutation/recombination rate, sample size, etc., commandXX.R scripts should be modified.
doXX.sh scripts will result in MS file format containing 198 sampled chromosomes.

### Summary Statistics and Images
In order to calculate summary statistics, user should run `sum_stats.py`.  
Sample size, sequence length, number of replicates, path to msms files, and the simulation type should be specified in `sum_stats.py` file.
This will generate XX.csv file for desired simulation scenario contaning summary statistics. 

`image_preprocess.py` can be used to create images from ms.txt files. 
Image size, flipping type, number of simulations, sample size and sequence length can also be specified by modifying this script. 

### Creating Deep Neural Network and Model Evaluation
__BaSe__ package is used in further analysis. It contains usefull functions for creating, evaluating and visualizing model. 

`ann.py` and  `cnn.py` scripts can be run to create regular neural network and convolutional neural network models, respectively. The test will be performed should be specified in scripts:

1. Neutral vs Selection(including IS, OD, and FD)
2. Incomplete Sweep(IS) vs Balancing selection(OD and FD)
3. Overdominance(OD) vs Negative Frequency Dependent Selection(FD) 

Model parameters can be changes from these scripts. 

#### Visualizing CNN
`cnn_vis.py` script can be used to visualize CNN results.
