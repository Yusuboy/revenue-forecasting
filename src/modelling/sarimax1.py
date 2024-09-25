import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the training dataset 
train_data = pd.read_csv('training_data.csv')


print(str(train_data.shape))
print(train_data)
print('----------------------')
# Load the validation dataset 
validation_data = pd.read_csv('validation_data.csv')
print(str(validation_data.shape))
print(validation_data)
print('----------------------')
# Remove currency symbols and convert to numeric in both datasets
train_data['Revenue FIN'] = train_data['Revenue FIN'].replace('[€,]', '', regex=True).astype(float)
train_data['Revenue IND'] = train_data['Revenue IND'].replace('[€,]', '', regex=True).astype(float)
train_data['Revenue NS'] = train_data['Revenue NS'].replace('[€,]', '', regex=True).astype(float)

validation_data['Revenue FIN'] = validation_data['Revenue FIN'].replace('[€,]', '', regex=True).astype(float)
validation_data['Revenue IND'] = validation_data['Revenue IND'].replace('[€,]', '', regex=True).astype(float)
validation_data['Revenue NS'] = validation_data['Revenue NS'].replace('[€,]', '', regex=True).astype(float)

# Create DateTime indices for proper time-series handling
train_data['Date'] = pd.to_datetime(train_data[['Year', 'Month']].assign(DAY=1))
train_data.set_index('Date', inplace=True)

validation_data['Date'] = pd.to_datetime(validation_data[['Year', 'Month']].assign(DAY=1))
validation_data.set_index('Date', inplace=True)

# Add a time trend variable (in months) to both datasets
train_data['time_trend'] = range(1, len(train_data) + 1)
validation_data['time_trend'] = range(len(train_data) + 1, len(train_data) + len(validation_data) + 1)

# Define the target variables for the training data
y_train_FIN = train_data['Revenue FIN']
y_train_IND = train_data['Revenue IND']
y_train_NS = train_data['Revenue NS']

# Define the exogenous variables (working days and time trend) for the training data
X_train_FIN = train_data[['D/M FIN', 'time_trend']]
X_train_IND = train_data[['D/M IND', 'time_trend']]
X_train_NS = train_data[['D/M NS', 'time_trend']]

# Validation target and exogenous data (April 2024 - August 2024)
y_valid_FIN = validation_data['Revenue FIN']
y_valid_IND = validation_data['Revenue IND']
y_valid_NS = validation_data['Revenue NS']

X_valid_FIN = validation_data[['D/M FIN', 'time_trend']]
X_valid_IND = validation_data[['D/M IND', 'time_trend']]
X_valid_NS = validation_data[['D/M NS', 'time_trend']]

# 1. Fit the SARIMAX model for Revenue FIN
model_FIN = sm.tsa.SARIMAX(endog=y_train_FIN, exog=X_train_FIN, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_FIN = model_FIN.fit()

# 2. Fit the SARIMAX model for Revenue IND
model_IND = sm.tsa.SARIMAX(endog=y_train_IND, exog=X_train_IND, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_IND = model_IND.fit()

# 3. Fit the SARIMAX model for Revenue NS
model_NS = sm.tsa.SARIMAX(endog=y_train_NS, exog=X_train_NS, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results_NS = model_NS.fit()

# Forecast on the validation set 
forecast_FIN = results_FIN.forecast(steps=len(y_valid_FIN), exog=X_valid_FIN)
forecast_IND = results_IND.forecast(steps=len(y_valid_IND), exog=X_valid_IND)
forecast_NS = results_NS.forecast(steps=len(y_valid_NS), exog=X_valid_NS)

# Combine the forecasts into total revenue forecast
forecast_total = forecast_FIN + forecast_IND + forecast_NS

# Print forecasted values in integer format
print("Forecasted Revenue FIN:")
print(forecast_FIN.apply(lambda x: f'{int(x):,d}'))

print("Forecasted Revenue IND:")
print(forecast_IND.apply(lambda x: f'{int(x):,d}'))

print("Forecasted Revenue NS:")
print(forecast_NS.apply(lambda x: f'{int(x):,d}'))

print("Total Forecasted Revenue:")
print(forecast_total.apply(lambda x: f'{int(x):,d}'))

# Calculate accuracy metrics (RMSE and MAE) for each component and format as integers
rmse_FIN = int(np.sqrt(mean_squared_error(y_valid_FIN, forecast_FIN)))
rmse_IND = int(np.sqrt(mean_squared_error(y_valid_IND, forecast_IND)))
rmse_NS = int(np.sqrt(mean_squared_error(y_valid_NS, forecast_NS)))
rmse_total = int(np.sqrt(mean_squared_error(y_valid_FIN + y_valid_IND + y_valid_NS, forecast_total)))

mae_FIN = int(mean_absolute_error(y_valid_FIN, forecast_FIN))
mae_IND = int(mean_absolute_error(y_valid_IND, forecast_IND))
mae_NS = int(mean_absolute_error(y_valid_NS, forecast_NS))
mae_total = int(mean_absolute_error(y_valid_FIN + y_valid_IND + y_valid_NS, forecast_total))

# Print RMSE and MAE for each component and total revenue as integers
print(f"RMSE for Revenue FIN: {rmse_FIN:,d}")
print(f"RMSE for Revenue IND: {rmse_IND:,d}")
print(f"RMSE for Revenue NS: {rmse_NS:,d}")
print(f"RMSE for Total Revenue: {rmse_total:,d}")

print(f"MAE for Revenue FIN: {mae_FIN:,d}")
print(f"MAE for Revenue IND: {mae_IND:,d}")
print(f"MAE for Revenue NS: {mae_NS:,d}")
print(f"MAE for Total Revenue: {mae_total:,d}")

# Print monthly forecasting errors in integer format
monthly_errors = pd.DataFrame({
    'Actual Revenue FIN': y_valid_FIN.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue FIN': forecast_FIN.apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue IND': y_valid_IND.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue IND': forecast_IND.apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue NS': y_valid_NS.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue NS': forecast_NS.apply(lambda x: f'{int(x):,d}'),
    'Actual Total Revenue': (y_valid_FIN + y_valid_IND + y_valid_NS).apply(lambda x: f'{int(x):,d}'),
    'Forecasted Total Revenue': forecast_total.apply(lambda x: f'{int(x):,d}')
})

print("\nMonthly Forecasting Errors (April 2024 - August 2024):")
print(monthly_errors)

# Combine the actual revenue from training and validation datasets for plotting
actual_total_train = y_train_FIN + y_train_IND + y_train_NS
actual_total_valid = y_valid_FIN + y_valid_IND + y_valid_NS

# Plot the actual vs forecast for total revenue
plt.figure(figsize=(10,6))
plt.plot(actual_total_train.index, actual_total_train, label='Actual Total Revenue (Training)', color='blue')
plt.plot(actual_total_valid.index, actual_total_valid, label='Actual Total Revenue (Validation)', color='blue', linestyle='--')

# Only plot forecasted revenue for the validation period
plt.plot(validation_data.index, forecast_total, label='Forecasted Total Revenue (Validation)', linestyle='--', color='red')

plt.title('Actual vs Forecasted Total Revenue (Oct 2021 - Aug 2024)')
plt.xlabel('Date')
plt.ylabel('Revenue (€)')
plt.legend()
plt.show()