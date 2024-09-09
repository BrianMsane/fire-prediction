
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
