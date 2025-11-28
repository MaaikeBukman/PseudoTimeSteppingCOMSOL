In this folder there are COMSOL and Python files for running COMSOL files 
with the predicted CFL number of the trained network. 

COMSOL files: There are several studies for computing converged solution, performing
the model with the CFL prediction and compute the solution after 1 iteration.

Python files: Make network predictions, run a loop until COMSOL solution has converged
rewrite COMSOL data to network input data, trained network


-----------------------------------------------------------------------------------
-------PYTHON files-------

%%RUNRESULTS.py%%
Runs the results for certain COMSOL file with different inflow velocities, 
mesh sizes and/or obstacle(s) (locations).
Writes results to excel file.
Runs Run_Network.m to obtain network results.


%%RUNNETWORK.py%%
Function that runs the network each iteration until convergence has been reached.
Calls MakePredictions.m to make a prediction each iteration.


%%MAKEPREDICTIONS.py%%
Makes a CFL prediction by running the network and generating data with Gen_data3.m.


%%GENERATEDATA.py%%
Generate data from COMSOL simulation.


%%BOUNDARYELEMENTS.py%%
Generate data for boundary elements in Gen_data3.m.


-------COMSOL files-------

%%BACKSTEP_GEN.MPH%%
Call this COMSOL file to call and use the back-step simulation

