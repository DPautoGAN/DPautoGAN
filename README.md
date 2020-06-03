# DPautoGAN

Code for the paper Differentially Private Mixed-Type Data Generation for Unsupervised Learning.

### Dependencies
```
certifi==2020.4.5.1
cycler==0.10.0
future==0.18.2
joblib==0.15.1
kiwisolver==1.2.0
matplotlib==3.2.1
numpy==1.18.4
pandas==1.0.4
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
scikit-learn==0.23.1
scipy==1.4.1
six==1.15.0
sklearn==0.0
threadpoolctl==2.1.0
torch==1.5.0
```
### UCI Data Generation

To run the DPautoGAN on UCI Dataset, follow the uci.ipynb jupyter notebook. Cell 2 requires the correct names of the dataset files to be passed in. If you have exactly cloned the repository and have not made any changes then this should not be an issue.

The rest of the notebook can be run as is.

If you train different models, with different hyperparameter values, make sure to change the filenames of the saved and loaded models to ensure consistency.

To perform the PCA evaluation on the generated data, follow the pca-eval.ipynb notebook (making sure to change the filename of the loaded models). To load the generated data into this file, you can either go back to uci.ipynb and save one of the generated values during the latter half of the notebook that is dedicated to evaluating the generated data or copy the code used to generate that data.



### Mimic Data Generation

In order to run the MIMIC data generation process:

1. Run the dp_autoencoder.py file. The parameters to be passed in are at the bottom of the file.
2. Next, run the dp_wgan.py file. Make sure that the autoencoder model loaded into the GAN is the same as the one you saved in the previous step.
3. Run evaluation.py. At the bottom of the file, you can choose to comment or un-comment out the evaluations you wish to run.



### Further evaluation

The above two data generation processes also show how to evaluate the data for ROC/AUC and PCA. 
In order to get the Wasserstein distance for the PCA plots:
1. We take a random sample of size 100 from real and synthetic PCA datasets. We compute the the matrix of pairwise distances and write it to a csv file. This process is repeated 100 times to create "i.csv" files where i varies from 0 to 99. 
2. These files are then used while running eval_wasserstein.R code. This code evaluates Wasserstein distance between the data points for each of the 100 files and outputs the average.  
The same process is used for computing Wasserstein distance on real and synthetic full datasets.
