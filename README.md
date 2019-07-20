# Balancing Selection

This repo contains the code used for distinguishing between balancing selection and incomplete sweep using deep learning.

### Simulations
We performed simulations for a neutral(NE) and three different selection scenarios, including incomplete sweep(IS), overdominance(OD) and negative frequency dependent selection(FD).
To run simulations, user should run doXX(NE/IS/OD/FD).sh scripts located in `simulation` folder. 
Different selection start times, dominance and selection coefficients can be set by modifying doXX.sh scripts. 
In order to change parameters of demographic model, mutation/recombination rate, sample size, etc., commandXX.R scripts should be modified.
The output file will be in MS file format containing 198 sampled chromosomes.

### BaSe
__BaSe__ package contains required modules for preprocessing simulation outputs and for creating, evaluating and visualizing models.

#### Summary Statistics and Images
In order to calculate summary statistics, user should run `sim_to_stats.py` script located in `run` folder.  
Sample size, sequence length, number of replicates (simulations), path to msms files, and the simulation type should be specified in this file.
This will generate XX.csv file for desired simulation scenario contaning summary statistics. 

`sim_to_image.py` can be used to create images from ms.txt files. 
Image size, number of simulations, sample size and sequence length can also be specified by modifying this script. 

#### Creating Deep Neural Network and Model Evaluation
`ann.py` and  `cnn.py` scripts can be run to create regular neural network and convolutional neural network models, respectively. The test will be performed should be specified in scripts:

1. Neutral vs Selection(including IS, OD, and FD)
2. Incomplete Sweep(IS) vs Balancing selection(OD and FD)
3. Overdominance(OD) vs Negative Frequency Dependent Selection(FD)  

##### Visualizing CNN
`cnn_vis.py` script can be used to visualize CNN results.
