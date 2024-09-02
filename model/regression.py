'''Take the challenge as regression
'''

import numpy as np
import pandas as pd
from utils import split, outlierHandler
from sklearn.metrics import mean_squared_error




def load_data(path: str):
    return  pd.read_csv(path)


def train(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    error = np.sqrt(mean_squared_error(y_true=y_test, y_pred=preds))
    return error


def create_model():
    model = None


def main(filename: str='submission.csv'):

    train = load_data(path='Train.csv')
    test = load_data(path='Test.csv')
    sub = load_data(path='SampleSubmission.csv')

    model = create_model()
    X_train, y_train, X_test, y_test = split(data=train)
    train(X_train, y_train, X_test, y_test, model)
    
    preds = model.predict(test)
    sub['burn_area'] = preds
    sub.to_csv(filename)


if __name__ == '__main__':
    main()
