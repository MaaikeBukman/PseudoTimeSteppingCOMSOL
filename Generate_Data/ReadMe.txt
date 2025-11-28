In this folder there are COMSOL, MATLAB and Excel files for generating data sets for 
the training of a ML model. 

COMSOL files: There are several studies for computing converged solution, computing
optimal CFL numbers and computing the residuals for the last nonlinear iteration.

MATLAB files: Adapt and run COMSOL files to obtain information.

Excel file: Data set used to train network.


-----------------------------------------------------------------------------------
-------MATLAB files-------

%%GEN_DATA3.M%%
Generates data point form one single element.
After that, GenPatchData3.m can be run in order to obtain patches.


%%GENPATCHDATA3.M%%
Creates patch data points with element data points obtained from Gen_data3.m


%%BOUNDARY_ELEMENTS.M%%
Creates boundary element point for GenPatchData3.m


%%CREATEDATASET.M%%
Combine different data sets created with GenPatchData3.m by randomly assigning
an equal amount of data point from each data set. 



-------COMSOL files-------

%%BACKSTEPOPT4C.MPH%%
Call this COMSOL file to call and use the back-step simulation for data generation


%%COUETTEOPT4C.MPH%%
Call this COMSOL file to call and use the Couette simulation for data generation


%%CAVITYOPT4C.MPH%%
Call this COMSOL file to call and use the cavity simulation for data generation



-------Excel file-------
%%DSTOT.XLSX%%
The data set used to train the network used for the thesis. It is a composition of
different simulations of 4 back-steps. 
