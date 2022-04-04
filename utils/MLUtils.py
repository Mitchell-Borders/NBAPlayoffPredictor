import pandas as pd
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.feature_selection import chi2


def get_input_output_split(csv):
    """Returns input (training) features and output (predicted) features after any necessary processing"""
    # read csv and split into input features (X) and output features (y)
    dataframe = pd.read_csv(csv)
    X = dataframe.drop(columns=['season', 'lg', 'team', 'abbreviation', 'g', 'playoffs'])
    y = dataframe['playoffs']

    # make classification labels Integers instead of Strings
    y_factorize = pd.Index.factorize(y)
    y = y_factorize[0]

    # Normalize data
    X = normalize_data(X)

    # Perform Feature Selection
    print(f'X before feature selection:{X.shape[1]}')
    X = feature_selection(X, y)
    print(f'X after feature selection:{X.shape}')

    return X, y


def feature_selection(X, y):
    # chi2 is what grades how good a feature is
    # mode/param go hand-in-hand as we take the top 80% of important features.
    transformer = sklearn.feature_selection.GenericUnivariateSelect(chi2, mode='percentile', param=80)
    X = transformer.fit_transform(X, y)
    return X

def normalize_data(X):
    normalizer = preprocessing.Normalizer().fit(X)
    X = normalizer.transform(X)
    return X




