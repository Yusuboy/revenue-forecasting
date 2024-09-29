import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Hours data summary.csv')

# Filter the data for the training period and validation period
train_data = data[(data['Year'] >= 2021) & (data['Month'] >= 9) & (data['Year'] <= 2023)]
validation_data = data[(data['Year'] >= 2024) & (data['Month'] >= 1) & (data['Month'] <= 8)]

# Ignore months 7-9 of 2021 and 9 of 2024
train_data = train_data[~((train_data['Year'] == 2021) & (train_data['Month'] < 9))]
validation_data = validation_data[validation_data['Month'] != 9]

# Create DateTime indices for proper time-series handling
train_data['Date'] = pd.to_datetime(train_data[['Year', 'Month']].assign(DAY=1))
train_data.set_index('Date', inplace=True)

validation_data['Date'] = pd.to_datetime(validation_data[['Year', 'Month']].assign(DAY=1))
validation_data.set_index('Date', inplace=True)

# Add a time trend variable (in months) to both datasets
train_data['time_trend'] = range(1, len(train_data) + 1)
validation_data['time_trend'] = range(len(train_data) + 1, len(train_data) + len(validation_data) + 1)

# Initialize dictionaries to store models, results, forecasts, and errors
sarimax_models = {}
sarimax_results = {}
forecasts = {}
rmse_values = {}
mae_values = {}

# Model 1: Revenue FIN, explaining variables: Hours 10* FI, Hours 20* FI, Hours 30*FI, D/M FIN
y_train_fin = train_data['Revenue FIN']
X_train_fin = train_data[['Hours 10* FI', 'Hours 20* FI', 'Hours 30*FI', 'D/M FIN', 'time_trend']]
y_valid_fin = validation_data['Revenue FIN']
X_valid_fin = validation_data[['Hours 10* FI', 'Hours 20* FI', 'Hours 30*FI', 'D/M FIN', 'time_trend']]

# Fit the SARIMAX model for Revenue FIN
sarimax_models['Revenue FIN'] = sm.tsa.SARIMAX(endog=y_train_fin, exog=X_train_fin, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results['Revenue FIN'] = sarimax_models['Revenue FIN'].fit()

# Forecast on the validation set for Revenue FIN
forecasts['Revenue FIN'] = sarimax_results['Revenue FIN'].forecast(steps=len(y_valid_fin), exog=X_valid_fin)

# Model 2: Revenue IND, explaining variables: Hours 20* IN, Hours 30*IN, D/M IND
y_train_ind = train_data['Revenue IND']
X_train_ind = train_data[['Hours 20* IN', 'Hours 30*IN', 'D/M IND', 'time_trend']]
y_valid_ind = validation_data['Revenue IND']
X_valid_ind = validation_data[['Hours 20* IN', 'Hours 30*IN', 'D/M IND', 'time_trend']]

# Fit the SARIMAX model for Revenue IND
sarimax_models['Revenue IND'] = sm.tsa.SARIMAX(endog=y_train_ind, exog=X_train_ind, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results['Revenue IND'] = sarimax_models['Revenue IND'].fit()

# Forecast on the validation set for Revenue IND
forecasts['Revenue IND'] = sarimax_results['Revenue IND'].forecast(steps=len(y_valid_ind), exog=X_valid_ind)

# Model 3: Revenue NS, explaining variables: Hours 20* NS, Hours 30*NS, D/M NS
y_train_ns = train_data['Revenue NS']
X_train_ns = train_data[['Hours 20* NS', 'Hours 30*NS', 'D/M NS', 'time_trend']]
y_valid_ns = validation_data['Revenue NS']
X_valid_ns = validation_data[['Hours 20* NS', 'Hours 30*NS', 'D/M NS', 'time_trend']]

# Fit the SARIMAX model for Revenue NS
sarimax_models['Revenue NS'] = sm.tsa.SARIMAX(endog=y_train_ns, exog=X_train_ns, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results['Revenue NS'] = sarimax_models['Revenue NS'].fit()

# Forecast on the validation set for Revenue NS
forecasts['Revenue NS'] = sarimax_results['Revenue NS'].forecast(steps=len(y_valid_ns), exog=X_valid_ns)

# Combine the forecasts into total revenue forecast (sum of all locations)
forecast_total = forecasts['Revenue FIN'] + forecasts['Revenue IND'] + forecasts['Revenue NS']

# Calculate RMSE and MAE for each model and for the total revenue
rmse_values['Revenue FIN'] = np.sqrt(mean_squared_error(y_valid_fin, forecasts['Revenue FIN']))
mae_values['Revenue FIN'] = mean_absolute_error(y_valid_fin, forecasts['Revenue FIN'])

rmse_values['Revenue IND'] = np.sqrt(mean_squared_error(y_valid_ind, forecasts['Revenue IND']))
mae_values['Revenue IND'] = mean_absolute_error(y_valid_ind, forecasts['Revenue IND'])

rmse_values['Revenue NS'] = np.sqrt(mean_squared_error(y_valid_ns, forecasts['Revenue NS']))
mae_values['Revenue NS'] = mean_absolute_error(y_valid_ns, forecasts['Revenue NS'])

rmse_total = np.sqrt(mean_squared_error(validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS'], forecast_total))
mae_total = mean_absolute_error(validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS'], forecast_total)

# Print RMSE and MAE for each model
print(f"RMSE for Revenue FIN: {rmse_values['Revenue FIN']}")
print(f"MAE for Revenue FIN: {mae_values['Revenue FIN']}")

print(f"RMSE for Revenue IND: {rmse_values['Revenue IND']}")
print(f"MAE for Revenue IND: {mae_values['Revenue IND']}")

print(f"RMSE for Revenue NS: {rmse_values['Revenue NS']}")
print(f"MAE for Revenue NS: {mae_values['Revenue NS']}")

print(f"RMSE for Total Revenue: {rmse_total}")
print(f"MAE for Total Revenue: {mae_total}")

# Plot the actual vs forecast for total revenue
plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data['Revenue FIN'] + train_data['Revenue IND'] + train_data['Revenue NS'], label='Actual Total Revenue (Training)', color='blue')
plt.plot(validation_data.index, validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS'], label='Actual Total Revenue (Validation)', color='blue', linestyle='--')

# Only plot forecasted revenue for the validation period
plt.plot(validation_data.index, forecast_total, label='Forecasted Total Revenue (Validation)', linestyle='--', color='red')

plt.title('Actual vs Forecasted Total Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue (€)')
plt.legend()
plt.show()