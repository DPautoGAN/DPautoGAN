import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score, explained_variance_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import Lasso

import torch
from torch import nn
import torch.nn.functional as F

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

            if 'categorical' in datatype:
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
    ('sex', 'categorical binary'),
    ('capital-gain', 'positive float'),
    ('capital-loss', 'positive float'),
    ('hours-per-week', 'positive int'),
    ('native-country', 'categorical'),
    ('salary', 'categorical binary'),
]

processor = Processor(datatypes)

relevant_df = df.drop(columns=['education', 'fnlwgt'])
for column, datatype in datatypes:
    if 'categorical' in datatype:
        relevant_df[column] = relevant_df[column].astype('category').cat.codes

train_df = relevant_df.head(32562)

X_real = torch.tensor(relevant_df.values.astype('float32'))
X_encoded = torch.tensor(processor.fit_transform(X_real).astype('float32'))

train_cutoff = 32562

X_train_real = X_real[:train_cutoff]
X_test_real = X_real[:train_cutoff]

X_train_encoded = X_encoded[:train_cutoff]
X_test_encoded = X_encoded[train_cutoff:]

print(X_encoded.shape)

print(X_train_encoded)
print(X_test_encoded)

ae_params = {
    'b1': 0.9,
    'b2': 0.999,
    'binary': False,
    'compress_dim': 15,
    'delta': 1e-5,
    'device': 'cuda',
    'iterations': 20000,
    'lr': 0.005,
    'l2_penalty': 0.,
    'l2_norm_clip': 0.012,
    'minibatch_size': 64,
    'microbatch_size': 1,
    'noise_multiplier': 2.5,
    'nonprivate': True,
}

autoencoder = Autoencoder(
    example_dim=len(X_train_encoded[0]),
    compression_dim=ae_params['compress_dim'],
    binary=ae_params['binary'],
    device=ae_params['device'],
)

decoder_optimizer = dp_optimizer.DPAdam(
    l2_norm_clip=ae_params['l2_norm_clip'],
    noise_multiplier=ae_params['noise_multiplier'],
    minibatch_size=ae_params['minibatch_size'],
    microbatch_size=ae_params['microbatch_size'],
    nonprivate=ae_params['nonprivate'],
    params=autoencoder.get_decoder().parameters(),
    lr=ae_params['lr'],
    betas=(ae_params['b1'], ae_params['b2']),
    weight_decay=ae_params['l2_penalty'],
)

encoder_optimizer = torch.optim.Adam(
    params=autoencoder.get_encoder().parameters(),
    lr=ae_params['lr'] * ae_params['microbatch_size'] / ae_params['minibatch_size'],
    betas=(ae_params['b1'], ae_params['b2']),
    weight_decay=ae_params['l2_penalty'],
)

weights, ds = [], []
for name, datatype in datatypes:
    if 'categorical' in datatype:
        num_values = len(np.unique(relevant_df[name]))
        if num_values == 2:
            weights.append(1.)
            ds.append((datatype, 1))
        else:
            for i in range(num_values):
                weights.append(1. / num_values)
            ds.append((datatype, num_values))
    else:
        weights.append(1.)
        ds.append((datatype, 1))

weights = torch.tensor(weights).to(ae_params['device'])

#autoencoder_loss = (lambda input, target: torch.mul(weights, torch.pow(input-target, 2)).sum(dim=1).mean(dim=0))
#autoencoder_loss = lambda input, target: torch.mul(weights, F.binary_cross_entropy(input, target, reduction='none')).sum(dim=1).mean(dim=0)
autoencoder_loss = nn.BCELoss()
#autoencoder_loss = nn.MSELoss()

print(autoencoder)

print('Achieves ({}, {})-DP'.format(
    analysis.epsilon(
        len(X_train_encoded),
        ae_params['minibatch_size'],
        ae_params['noise_multiplier'],
        ae_params['iterations'],
        ae_params['delta']
    ),
    ae_params['delta'],
))

minibatch_loader, microbatch_loader = sampling.get_data_loaders(
    minibatch_size=ae_params['minibatch_size'],
    microbatch_size=ae_params['microbatch_size'],
    iterations=ae_params['iterations'],
    nonprivate=ae_params['nonprivate'],
)

train_losses, validation_losses = [], []

X_train_encoded = X_train_encoded.to(ae_params['device'])
X_test_encoded = X_test_encoded.to(ae_params['device'])

for iteration, X_minibatch in enumerate(minibatch_loader(X_train_encoded)):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    for X_microbatch in microbatch_loader(X_minibatch):

        decoder_optimizer.zero_microbatch_grad()
        output = autoencoder(X_microbatch)
        loss = autoencoder_loss(output, X_microbatch)
        loss.backward()
        decoder_optimizer.microbatch_step()

    validation_loss = autoencoder_loss(autoencoder(X_test_encoded).detach(), X_test_encoded)

    encoder_optimizer.step()
    decoder_optimizer.step()

    train_losses.append(loss.item())
    validation_losses.append(validation_loss.item())

    if iteration % 1000 == 0:
        print ('[Iteration %d/%d] [Loss: %f] [Validation Loss: %f]' % (
            iteration, ae_params['iterations'], loss.item(), validation_loss.item())
        )

pd.DataFrame(data={'train': train_losses, 'validation': validation_losses}).plot()

with open('ae_eps_inf.dat', 'wb') as f:
    torch.save(autoencoder, f)

gan_params = {
    'alpha': 0.99,
    'binary': False,
    'clip_value': 0.01,
    'd_updates': 15,
    'delta': 1e-5,
    'device': 'cuda',
    'iterations': 15000,
    'latent_dim': 64,
    'lr': 0.005,
    'l2_penalty': 0.,
    'l2_norm_clip': 0.022,
    'minibatch_size': 128,
    'microbatch_size': 1,
    'noise_multiplier': 3.5,
    'nonprivate': False,
}

with open('ae_eps_inf.dat', 'rb') as f:
    autoencoder = torch.load(f)
decoder = autoencoder.get_decoder()

generator = Generator(
    input_dim=gan_params['latent_dim'],
    output_dim=autoencoder.get_compression_dim(),
    binary=gan_params['binary'],
    device=gan_params['device'],
)

g_optimizer = torch.optim.RMSprop(
    params=generator.parameters(),
    lr=gan_params['lr'],
    alpha=gan_params['alpha'],
    weight_decay=gan_params['l2_penalty'],
)

discriminator = Discriminator(
    input_dim=len(X_train_encoded[0]),
    device=gan_params['device'],
)

d_optimizer = dp_optimizer.DPRMSprop(
    l2_norm_clip=gan_params['l2_norm_clip'],
    noise_multiplier=gan_params['noise_multiplier'],
    minibatch_size=gan_params['minibatch_size'],
    microbatch_size=gan_params['microbatch_size'],
    nonprivate=gan_params['nonprivate'],
    params=discriminator.parameters(),
    lr=gan_params['lr'],
    alpha=gan_params['alpha'],
    weight_decay=gan_params['l2_penalty'],
)

print(generator)
print(discriminator)

print('Achieves ({}, {})-DP'.format(
    analysis.epsilon(
        len(X_train_encoded),
        gan_params['minibatch_size'],
        gan_params['noise_multiplier'],
        gan_params['iterations'],
        gan_params['delta']
    ),
    gan_params['delta'],
))

minibatch_loader, microbatch_loader = sampling.get_data_loaders(
    minibatch_size=gan_params['minibatch_size'],
    microbatch_size=gan_params['microbatch_size'],
    iterations=gan_params['iterations'],
    nonprivate=gan_params['nonprivate'],
)

X_train_encoded = X_train_encoded.to(gan_params['device'])
X_test_encoded = X_test_encoded.to(ae_params['device'])

for iteration, X_minibatch in enumerate(minibatch_loader(X_train_encoded)):

    d_optimizer.zero_grad()

    for real in microbatch_loader(X_minibatch):
        z = torch.randn(real.size(0), gan_params['latent_dim'], device=gan_params['device'])
        fake = decoder(generator(z)).detach()

        d_optimizer.zero_microbatch_grad()
        d_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
        d_loss.backward()
        d_optimizer.microbatch_step()

    d_optimizer.step()

    for parameter in discriminator.parameters():
        parameter.data.clamp_(-gan_params['clip_value'], gan_params['clip_value'])

    if iteration % gan_params['d_updates'] == 0:
        z = torch.randn(X_minibatch.size(0), gan_params['latent_dim'], device=gan_params['device'])
        fake = decoder(generator(z))

        g_optimizer.zero_grad()
        g_loss = -torch.mean(discriminator(fake))
        g_loss.backward()
        g_optimizer.step()

    if iteration % 1000 == 0:
        print('[Iteration %d/%d] [D loss: %f] [G loss: %f]' % (
            iteration, gan_params['iterations'], d_loss.item(), g_loss.item()
        ))

        z = torch.randn(len(X_train_real), gan_params['latent_dim'], device=gan_params['device'])
        X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()
        X_synthetic_real = processor.inverse_transform(X_synthetic_encoded)
        X_synthetic_encoded = processor.transform(X_synthetic_real)
        synthetic_data = pd.DataFrame(X_synthetic_real, columns=relevant_df.columns)

        i = 0
        columns = relevant_df.columns
        relevant_df[columns[i]].hist()
        synthetic_data[columns[i]].hist()
        plt.show()

        #pca_evaluation(pd.DataFrame(X_train_real), pd.DataFrame(X_synthetic_real))
        #plt.show()

with open('gen_eps_inf.dat', 'wb') as f:
    torch.save(generator, f)

X_train_encoded = X_train_encoded.cpu()
X_test_encoded = X_test_encoded.cpu()

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_encoded[:,:-1], X_train_encoded[:,-1])
prediction = clf.predict(X_test_encoded[:,:-1])

print(accuracy_score(X_test_encoded[:,-1], prediction))
print(f1_score(X_test_encoded[:,-1], prediction))

with open('gen_eps_inf.dat', 'rb') as f:
    generator = torch.load(f)

with open('ae_eps_inf.dat', 'rb') as f:
    autoencoder = torch.load(f)
decoder = autoencoder.get_decoder()

z = torch.randn(len(X_train_real), gan_params['latent_dim'], device=gan_params['device'])
X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()
X_synthetic_real = processor.inverse_transform(X_synthetic_encoded)
X_synthetic_encoded = processor.transform(X_synthetic_real)

#pd.DataFrame(X_encoded.numpy()).to_csv('real.csv')
pd.DataFrame(X_synthetic_encoded).to_csv('synthetic.csv')

with open('gen_eps_inf.dat', 'rb') as f:
    generator = torch.load(f)

with open('ae_eps_inf.dat', 'rb') as f:
    autoencoder = torch.load(f)
decoder = autoencoder.get_decoder()

X_test_encoded = X_test_encoded.cpu()

z = torch.randn(len(X_train_real), gan_params['latent_dim'], device=gan_params['device'])
X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()

X_synthetic_real = processor.inverse_transform(X_synthetic_encoded)
X_synthetic_encoded = processor.transform(X_synthetic_real)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_synthetic_encoded[:,:-1], X_synthetic_encoded[:,-1])
prediction = clf.predict(X_test_encoded[:,:-1])

print(accuracy_score(X_test_encoded[:,-1], prediction))
print(f1_score(X_test_encoded[:,-1], prediction))

with open('gen_eps_inf.dat', 'rb') as f:
    generator = torch.load(f)

with open('ae_eps_inf.dat', 'rb') as f:
    autoencoder = torch.load(f)
decoder = autoencoder.get_decoder()

z = torch.randn(len(X_train_real), gan_params['latent_dim'], device=gan_params['device'])
X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()
X_synthetic_real = processor.inverse_transform(X_synthetic_encoded)
synthetic_data = pd.DataFrame(X_synthetic_real, columns=relevant_df.columns)

column = 'age'
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(train_df[column].values,)# bins=)
ax.hist(synthetic_data[column].values, color='red', alpha=0.35,)# bins10)

with open('gen_eps_inf.dat', 'rb') as f:
    generator = torch.load(f)

with open('ae_eps_inf.dat', 'rb') as f:
    autoencoder = torch.load(f)
decoder = autoencoder.get_decoder()

z = torch.randn(len(X_train_real), gan_params['latent_dim'], device=gan_params['device'])
X_synthetic_encoded = decoder(generator(z)).cpu().detach().numpy()
X_synthetic_real = processor.inverse_transform(X_synthetic_encoded)
synthetic_data = pd.DataFrame(X_synthetic_real, columns=relevant_df.columns)

regression_real = []
classification_real = []
regression_synthetic = []
classification_synthetic = []
target_real = []
target_synthetic = []

for column, datatype in datatypes:
    p = Processor([datatype for datatype in datatypes if datatype[0] != column])

    train_cutoff = 32562

    p.fit(relevant_df.drop(columns=[column]).values)

    X_enc = p.transform(relevant_df.drop(columns=[column]).values)
    y_enc = relevant_df[column]

    X_enc_train = X_enc[:train_cutoff]
    X_enc_test = X_enc[train_cutoff:]

    y_enc_train = y_enc[:train_cutoff]
    y_enc_test = y_enc[train_cutoff:]

    X_enc_syn = p.transform(synthetic_data.drop(columns=[column]).values)
    y_enc_syn = synthetic_data[column]

    if 'binary' in datatype:
        model = lambda: RandomForestClassifier(n_estimators=10)
        score = lambda true, pred: f1_score(true, pred)
    elif 'categorical' in datatype:
        model = lambda: RandomForestClassifier(n_estimators=10)
        score = lambda true, pred: f1_score(true, pred, average='micro')
    else:
        model = lambda: Lasso()
        explained_var = lambda true, pred: explained_variance_score(true, pred)
        score = r2_score

    real, synthetic = model(), model()

    real.fit(X_enc_train, y_enc_train)
    synthetic.fit(X_enc_syn, y_enc_syn)

    real_preds = real.predict(X_enc_test)
    synthetic_preds = synthetic.predict(X_enc_test)

    print(column, datatype)
    if column == 'salary':
        target_real.append(score(y_enc_test, real_preds))
        target_synthetic.append(score(y_enc_test, synthetic_preds))
    elif 'categorical' in datatype:
        classification_real.append(score(y_enc_test, real_preds))
        classification_synthetic.append(score(y_enc_test, synthetic_preds))
    else:
        regression_real.append(score(y_enc_test, real_preds))
        regression_synthetic.append(score(y_enc_test, synthetic_preds))

    print(score.__name__)
    print('Real: {}'.format(score(y_enc_test, real_preds)))
    print('Synthetic: {}'.format(score(y_enc_test, synthetic_preds)))
    print('')

plt.scatter(classification_real, classification_synthetic, c='blue')
plt.scatter(regression_real, regression_synthetic, c='red')
plt.scatter(target_real, target_synthetic, c='green')
plt.xlabel('Real Data')
plt.ylabel('Synthetic Data')
plt.axis((0., 1., 0., 1.))
plt.plot((0, 1), (0, 1))
plt.show()
