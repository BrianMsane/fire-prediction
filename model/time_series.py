'''Take the challenge as a time-series
'''

import typing
import datetime
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from prophet.forecaster import Prophet # type: ignore
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from statsmodels.tsa.holtwinters import ExponentialSmoothing # type: ignore
from statsmodels.nonparametric.smoothers_lowess import lowess # type: ignore
from pykalman import KalmanFilter # type: ignore
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft, ifft
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore
from pykalman import KalmanFilter # type: ignore
from pmdarima import auto_arima # type: ignore
from sklearn.metrics import mean_squared_error


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
    return data


# def train_sarimax_auto_arima(
#     train: pd.DataFrame, 
#     test: pd.DataFrame, 
#     non_seasonal_order=(2, 0, 0), 
#     best_seasonal_order=(2, 0, 1, 12)
# ):
#     rmse_values = []

#     for real_id in train['real_id'].unique():
#         # Filter the train and test data for the current area ID
#         train_area = train.loc[train.real_id == real_id]
#         test_area = test.loc[test.real_id == real_id]

#         # Train Test Split
#         X_train_area = train_area.loc[train_area.index.strftime('%Y-%m-%d') < '2011-01-01']
#         y_test_area = train_area.loc[train_area.index.strftime('%Y-%m-%d') >= '2011-01-01']

#         try:
#             # Fit SARIMAX model with the best orders on X_train set
#             sarimax_model = SARIMAX(X_train_area['burn_area'],
#                                     order=non_seasonal_order,
#                                     seasonal_order=best_seasonal_order)
#             sarima_fit = sarimax_model.fit(disp=False)

#             # Forecast for the specific number of months
#             forecast_months = len(y_test_area)

#             # Perform the forecast
#             sarimax_pred = sarima_fit.get_forecast(steps=forecast_months)

#             # Extract predicted values
#             predicted_values = sarimax_pred.predicted_mean

#             # Apply Kalman Filter to smooth the predictions
#             kf = KalmanFilter(initial_state_mean=predicted_values.iloc[0], n_dim_obs=1)
#             predicted_values_kf, _ = kf.filter(predicted_values.values)
#             predicted_data_df = pd.Series(predicted_values_kf.flatten(), index=y_test_area.index, name='burn_area')

#         except Exception as e:
#             print(f"Failed to model real_id {real_id}: {e}. Generating ARIMA forecast")
#             try:
#                 # Auto ARIMA model tuning
#                 model_arima = auto_arima(X_train_area['burn_area'], m=12, seasonal=True,
#                                          start_p=0, start_q=0, max_order=5, test='adf', error_action='ignore',
#                                          suppress_warnings=True, stepwise=True, trace=False)
#                 forecast_length = len(y_test_area)
#                 predicted_values = model_arima.predict(n_periods=forecast_length)

#                 # Apply Kalman Filter to smooth the predictions
#                 kf = KalmanFilter(initial_state_mean=predicted_values[0], n_dim_obs=1)
#                 predicted_values_kf, _ = kf.filter(predicted_values)
#                 predicted_data_df = pd.Series(predicted_values_kf.flatten(), index=test_area.index, name='burn_area')

#             except Exception as e:
#                 print(f"Failed to model real_id {real_id}: {e}. Falling back to zero prediction.")
#                 # Fall back to simple zero prediction
#                 predicted_data_df = pd.Series(np.zeros(len(y_test_area)), index=y_test_area.index)

#         # Ensure alignment
#         predicted_data_df = predicted_data_df.reindex(y_test_area.index, method='nearest')

#         # Calculate RMSE for the current area ID
#         rmse_area = np.sqrt(mean_squared_error(y_test_area['burn_area'], predicted_data_df))
#         rmse_values.append(rmse_area)

#     # Calculate the overall RMSE by averaging the RMSE values of all areas
#     if rmse_values:
#         overall_rmse = np.mean(rmse_values)
#         print('Overall Test RMSE:', overall_rmse)
#     else:
#         print('No valid RMSE values were calculated')

#     return overall_rmse if rmse_values else None



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
