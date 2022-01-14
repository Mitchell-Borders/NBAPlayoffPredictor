import pandas as pd
from sklearn import preprocessing


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
    normalizer = preprocessing.Normalizer().fit(X)
    X = normalizer.transform(X)

    return X, y


