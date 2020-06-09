import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer, StandardScaler

import torch
from torch import nn

from dp_wgan import Generator, Discriminator
from dp_autoencoder import Autoencoder
from evaluation import *
import dp_optimizer, sampling, analysis, evaluation

torch.manual_seed(0)
np.random.seed(0)

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
train = pd.read_csv('adult.data', names=names)
test = pd.read_csv('adult.test', names=names)

df = pd.concat([train, test])

print(df)

class Processor:
    def __init__(self, datatypes):
        self.datatypes = datatypes

    def fit(self, matrix):
        preprocessors, cutoffs = [], []
        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)

            if 'categorical' in datatype:
                preprocessor = LabelBinarizer()
            else:
                preprocessor = MinMaxScaler()

            preprocessed_col = preprocessor.fit_transform(preprocessed_col)
            cutoffs.append(preprocessed_col.shape[1])
            preprocessors.append(preprocessor)

        self.cutoffs = cutoffs
        self.preprocessors = preprocessors

    def transform(self, matrix):
        preprocessed_cols = []

        for i, (column, datatype) in enumerate(self.datatypes):
            preprocessed_col = matrix[:,i].reshape(-1, 1)
            preprocessed_col = self.preprocessors[i].transform(preprocessed_col)
            preprocessed_cols.append(preprocessed_col)

        return np.concatenate(preprocessed_cols, axis=1)


    def fit_transform(self, matrix):
        self.fit(matrix)
        return self.transform(matrix)

    def inverse_transform(self, matrix):
        postprocessed_cols = []

        j = 0
        for i, (column, datatype) in enumerate(self.datatypes):
            postprocessed_col = self.preprocessors[i].inverse_transform(matrix[:,j:j+self.cutoffs[i]])

            if datatype == 'categorical':
                postprocessed_col = postprocessed_col.reshape(-1, 1)
            else:
                if 'positive' in datatype:
                    postprocessed_col = postprocessed_col.clip(min=0)

                if 'int' in datatype:
                    postprocessed_col = postprocessed_col.round()

            postprocessed_cols.append(postprocessed_col)

            j += self.cutoffs[i]

        return np.concatenate(postprocessed_cols, axis=1)


datatypes = [
    ('age', 'positive int'),
    ('workclass', 'categorical'),
    ('education-num', 'categorical'),
    ('marital-status', 'categorical'),
    ('occupation', 'categorical'),
    ('relationship', 'categorical'),
    ('race', 'categorical'),
    ('sex', 'categorical'),
    ('capital-gain', 'positive float'),
    ('capital-loss', 'positive float'),
    ('hours-per-week', 'positive int'),
    ('native-country', 'categorical'),
    ('salary', 'categorical'),
]

processor = Processor(datatypes)

relevant_df = df.drop(columns=['education', 'fnlwgt'])
for column, datatype in datatypes:
    if datatype == 'categorical':
        relevant_df[column] = relevant_df[column].astype('category').cat.codes

X_real = torch.tensor(relevant_df.to_numpy(dtype=np.float32).astype('float32'))
X_encoded = torch.tensor(processor.fit_transform(X_real).astype('float32'))

train_cutoff = 32562

X_train_real = X_real[:train_cutoff]
X_test_real = X_real[:train_cutoff]

X_train_encoded = X_encoded[:train_cutoff]
X_test_encoded = X_encoded[train_cutoff:]

print(X_train_encoded)
print(X_test_encoded)

X_synthetic_encoded = pd.read_csv('chris_synth.csv').drop(columns=['Unnamed: 0']).to_numpy()

def std_PCA(W,d):
    '''
    Given a n x n matrix (this is equal to A^T A for an m x n data matrix A), output the top d eigenvectors of W as numpy n x d matrix.
    '''
    [eigenValues,eigenVectors] = np.linalg.eig(W)

    #sort eigenvalues and eigenvectors in decending orders
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    #take the first d vectors. Obtained the solution
    return eigenVectors[:,:d]

mask = np.random.choice([False, True], X_train_encoded.numpy().shape[0], p=[0.17, 0.83])

pca_basis = pd.DataFrame(X_train_encoded.numpy()[mask])
pca_real = ((pca_basis - pca_basis.mean())/pca_basis.std()).fillna(0)

pca_synth = ((pd.DataFrame(X_synthetic_encoded) - pca_basis.mean())/pca_basis.std()).fillna(0)

eigen_vectors = std_PCA((pca_real.to_numpy().T @ pca_real.to_numpy()), 2)

proj_real_real = pd.DataFrame(np.matmul(pca_real.to_numpy(), eigen_vectors))
proj_synth_real = pd.DataFrame(np.matmul(pca_synth.to_numpy(), eigen_vectors))

proj_real_real.to_csv('')
proj_real_real.plot(x = 0, y = 1, c ='red', kind = 'scatter', s =0.5)

proj_synth_real.to_csv('')
proj_synth_real.plot(x = 0, y = 1, c ='red', kind = 'scatter', s =0.5)
