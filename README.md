# BaSe
__BaSe__ (abbreviation for **Ba**lancing **Se**lection ) is a supervised machine learning algorithm
to predict selection and distinguish it between incomplete sweep and balancing selection. It includes
3 sequantial tests:

* __Test 1__: Neutrality vs Selection
* __Test 2__: Incomplete sweep vs Balancing selection
* __Test 3__: Overdominance vs Negative frequency dependent selection 

Each test performs a binary classification. The first test is to determine if the target allele is
under selection. Here, the selection class is a mixture of both balancing and positive selection
scenarios. If the target allele is under selection, the second test is to determine whether it is 
balancing selection or incomplete sweep. If it is a balancing selection, the third test distinguishes
different types of balancing selection: overdominance and negative frequency dependent selection.

BaSe implements both regular artifical neural network (ANN) and convolutional neural network (CNN).

### Usage
Please look at the `Simulations/README.md` for a quick tutorial on how to generate simulations. 
The jupyter notebooks `Tutorials/ANN_training.ipybn` and `Tutorials/CNN_training.ipybn` 
provide examples on how to use BaSe to train ANN and CNN, respectively. Then, the jupyter
notebook `Tutorials/Prediction.ipybn` demonstrate how to use trained models for prediction on
real data.

