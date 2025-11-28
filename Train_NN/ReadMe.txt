In this folder there are Python files for training a neural network 
with the data generated in the folder Generate_datasets


Python files: Create and train a neural network 

Excel files: Data generated from Generate_datasets

-----------------------------------------------------------------------------------
-------PYTHON files-------

%%MAIN.py%%
Select data set, #layers, #neurons, learning rate
Train network with TrainNeuralnet.py
Save trained network and corresponding normalization parameters

%%TRAINNEURALNET.py%%
Train Neural network
Decide on validation patience, #Epochs, MiniBatchSize, ValidationFrequency

%%GIVEDATA.py%%
Create train, validation and test data

%%NEURALNETWORK.py%%
Build neural network with given #layers and #neurons

