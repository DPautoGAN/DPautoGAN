from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dp_wgan import Generator


# Deterministic output
torch.manual_seed(0)
np.random.seed(0)


def isolate_column(df, column):
    """Splits a given dataframe into a column and the rest of the dataframe."""
    return df.drop(columns=[column]).values, df[column].values


def get_prediction_score(X_train, y_train, X_test, y_test, Model, score):
    """Trains a model on training data and returns the quality of the predictions given test data and expected output."""
    try:
        # Train model on synthetic data, see how well it predicts holdout real data
        clf = Model()
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
    except ValueError as e:
        # Only one label present, so simply predict all examples as that label
        prediction = np.full(y_test.shape, y_train[0])
    return score(y_test, prediction)

def feature_prediction_evaluation(
        train,
        test,
        synthetic,
        Model=lambda: LogisticRegression(solver='lbfgs'),
        score=lambda y_true, y_pred: f1_score(y_true, y_pred),
    ):
    """Runs the evaluation method given real and fake data.

    real: pd.DataFrame
    synthetic: pd.DataFrame, with columns identical to real
    Model: func, takes no arguments and returns object with fit and predict functions implemented, e.g. sklearn.linear_model.LogisticRegression
    score: func, takes in a ground truth and a prediction and returns the score of the prediction
    """
    if set(train.columns) != set(synthetic.columns):
        raise Exception('Columns of given datasets are not identical.')

    real_classifier_scores, synthetic_classifier_scores = [], []
    for i, column in enumerate(train.columns):
        X_train_real, y_train_real = isolate_column(train, column)
        X_train_synthetic, y_train_synthetic = isolate_column(synthetic, column)
        X_test, y_test = isolate_column(test, column)

        real_score = get_prediction_score(X_train_real, y_train_real, X_test, y_test, Model, score)
        synthetic_score = get_prediction_score(X_train_synthetic, y_train_synthetic, X_test, y_test, Model, score)

        real_classifier_scores.append(real_score)
        synthetic_classifier_scores.append(synthetic_score)
        print(i, real_score, synthetic_score)

    plt.scatter(real_classifier_scores, synthetic_classifier_scores, s=3, c='red')
    plt.title('Synthetic Data Evaluation Metric')
    plt.xlabel('Real Data')
    plt.ylabel('Synthetic Data')
    ax = plt.savefig('out.png')
    pd.DataFrame(data={'real': real_classifier_scores, 'synthetic': synthetic_classifier_scores}).to_csv('out.csv')

    return sum(map(lambda pair: (pair[0] - pair[1]) ** 2, zip(real_classifier_scores, synthetic_classifier_scores))), ax


def pca_evaluation(real, synthetic):
    pca = PCA(n_components=2)
    pca.fit(real.values)

    real_projection = pca.transform(real.values)
    synthetic_projection = pca.transform(synthetic.values)

    ax = pd.DataFrame(data=real_projection).plot(x=0, y=1, c='red', kind='scatter', s=0.5)
    ax = pd.DataFrame(data=synthetic_projection).plot(x=0, y=1, c='blue', kind='scatter', s=0.5, ax=ax)
    plt.savefig('out.png')

    return ax

