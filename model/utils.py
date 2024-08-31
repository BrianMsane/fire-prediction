

import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE


def scaler(X_train, X_test, minmax: bool=False):
    stdscaler = StandardScaler()
    X_train = stdscaler.fit_transform(X_train)
    X_test = stdscaler.transform(X_test)

    if minmax:
        minmaxscaler = MinMaxScaler(feature_range=(0,1))
        X_train = minmaxscaler.fit_transform(X_train)
        X_test = minmaxscaler.transform(X_test)
    return X_train, X_test


def handle_imbalance(X_train, y_train) -> tuple:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train


def reduce_dimensions(train: pd.DataFrame, n: int=2):
    pca = PCA(n_components=n)
    pca.fit(train)


def split(data, date: str='2011-01-01'):
    train = data.loc[data.date < date]
    valid = data.loc[data.date > date]

    target_col = 'burn_area'
    in_cols = [
        'climate_aet', 'climate_def',
        'climate_pdsi', 'climate_pet', 'climate_pr', 'climate_ro',
        'climate_soil', 'climate_srad', 'climate_swe', 'climate_tmmn',
        'climate_tmmx', 'climate_vap', 'climate_vpd', 'climate_vs', 'elevation',
        'landcover_0', 'landcover_1', 'landcover_2', 'landcover_3',
        'landcover_4', 'landcover_5', 'landcover_6', 'landcover_7',
        'landcover_8', 'precipitation'
    ]

    X_train, y_train = train[in_cols], train[target_col]
    X_test, y_test = valid[in_cols], valid[target_col]
    return X_train, y_train, X_test, y_test


def outlierHandler(data: pd.DataFrame) -> pd.DataFrame:
    numerics = []
    for col in data.columns:
        if data[col].dtype == 'float':
            numerics.append(col)

    for col in numerics:
        q1 = data[col].quantile(0.25)
        q2 = data[col].quantile(0.75)
        iqr = q2 - q1
        max_limit = q2 + (1.5 * iqr)
        min_limit = q1 - (1.5 * iqr)
        data[col]  = pd.DataFrame(
            np.where(data[col] > max_limit, max_limit, 
            (np.where(data[col] < min_limit, min_limit, data[col]))), columns=[col]
        )
    return data


def feature_selection(X_train, X_test, y_train, index: int) -> tuple:
    sel = RFE(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        n_features_to_select=index
    )
    sel.fit(X_train, y_train)
    X_train_rfe = sel.transform(X_train) 
    X_test_rfe = sel.transform(X_test)
    return X_train_rfe, X_test_rfe
