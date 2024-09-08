'''Take the challenge as a time-series
'''

import typing
import datetime
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from prophet import Prophet # type: ignore
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess # type: ignore
from pykalman import KalmanFilter # type: ignore
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft, ifft


def smooth_data(
    data: pd.DataFrame,
    algorithm: typing.Literal['SMA', 'WMA', 'SES', 'DES', 'TES', 'Kalman', 'LOESS', 'Gaussian', 'Fourier']='SMA',
    both: bool=False
):
    def apply_smoothing(series, method):
        if method == 'SMA':
            return series.rolling(window=5).mean()
        if method == 'WMA':
            weights = np.arange(1, 6)
            return series.rolling(window=5).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        if method == 'SES':
            return ExponentialSmoothing(series, trend=None, seasonal=None).fit().fittedvalues
        if method == 'DES':
            return ExponentialSmoothing(series, trend='add', seasonal=None).fit().fittedvalues
        if method == 'TES':
            return ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit().fittedvalues
        if method == 'Kalman':
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            state_means, _ = kf.filter(series.values)
            return pd.Series(state_means.flatten(), index=series.index)
        if method == 'LOESS':
            return pd.Series(lowess(series, np.arange(len(series)), frac=0.1)[:, 1], index=series.index)
        if method == 'Savitzky-Golay':
            return pd.Series(savgol_filter(series, window_length=5, polyorder=2), index=series.index)
        if method == 'Gaussian':
            return pd.Series(gaussian_filter1d(series, sigma=2), index=series.index)
        if method == 'Fourier':
            fft_series = fft(series)
            fft_series[int(len(fft_series)/10):] = 0
            return pd.Series(np.real(ifft(fft_series)), index=series.index)

    if both:
        smoothed_data = data.copy()
        for column in data.columns:
            smoothed_data[column] = apply_smoothing(data[column], algorithm)
        return smoothed_data
    else:
        return apply_smoothing(data, algorithm)



def add_regressors(data: pd.DataFrame):
    regressors = []
    vital = ('ds', 'y')
    for column in data.columns:
        if column not in vital:
            regressors.append(column)
    return regressors


def feature_engineering(data: pd.DataFrame):
    data['']


def train_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sub: pd.DataFrame,
    model : typing.Literal['prophet', 'arima']='prophet'
):
    if model == 'prophet':
        
        # Prophet format
        train['date'] = pd.to_datetime(train['ID'].apply(lambda x: x.split('_')[1]))
        train = train.sort_values(by='date')
        train['ds'] = train['date']
        train['y'] = train['burn_area']
        regressors = add_regressors(data=train)

        test['date'] = pd.to_datetime(test['ID'].apply(lambda x: x.split('_')[1]))
        test = test.sort_values(by='date')
        test['ds'] = test['date']

        train_all = train.copy().dropna()
        train = train_all.loc[train_all.ds < '2012-01-01']
        valid = train_all.loc[train_all.ds >= '2012-01-01']

        model = Prophet(
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None,
            scaling='absmax',
            holidays_mode=None,
        )
        for reg in regressors:
            model.add_regressor(name=reg, standardize=True)

        model.fit(train[['ds', 'y']] + regressors)
        preds = model.predict(valid['ds'])['yhat']
        print(f"RMSE: {np.sqrt(mean_squared_error(y_pred=preds, y_true=valid['y']))}")


        predictions = model.predict(test[['ds']] + regressors)
        sub['burn_area'] = predictions['yhat'].clip(0, 1)
        today = datetime.date.today()
        value = '02'
        sub.to_csv(f'submit_{today}_{value}')
    
    elif model == 'arima':
        model = None
    

def main():
    train = pd.read_csv('../data/Train.csv')
    test = pd.read_csv('../data/Test.csv')
    sub = pd.read_csv('../data/SampleSubmission.csv')

    train = feature_engineering(data=train)
    test = feature_engineering(data=test)

    train_model(
        train=train,
        test=test,
        sub=sub,
        model='prophet'
    )

if __name__ == '__main__':
    main()
