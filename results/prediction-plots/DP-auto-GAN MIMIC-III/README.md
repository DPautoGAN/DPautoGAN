This folder contains probability and prediction scores of each of the 1071 features of the MIMIC-III dataset. 
Noise multiplier of 3.5, 2.3, 1.3 are used to generate eps=0.81,1.33,2.70 settings.

Each folder contains the .csv of all probability and prediction scores. Prediction scores which AUROC is not defined is not included, but the column index retains information on which feature of MIMIC-III have defined and undefined AUROC.

The plots are named by setting a cut threshold of p. This means to plot only features with at least p fraction of 's presented in the original dataset, or equivalently the probability of that feature in prob.csv in the original (or real) data is at least p. 
Therefore, p=0 means all features are plotted. As p increases, less number of features are plotted, and only features with more stable (and arguably more meaningful) AUROC scores are retained.