import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the training dataset 
train_data = pd.read_csv('training_data_2.csv')
print(str(train_data.shape))
print(train_data)
print('----------------------')

# Load the validation dataset 
validation_data = pd.read_csv('validation_data_2.csv')
print(str(validation_data.shape))
print(validation_data)
print('----------------------')

# Remove currency symbols and convert to numeric in both datasets
location_columns = ['Revenue EU', 'Revenue FI 1', 'Revenue FI 2', 'Revenue IN 1', 'Revenue IN 2', 'Revenue IN 3', 'Revenue IN 4', 'Revenue PL']

for column in location_columns:
    train_data[column] = train_data[column].replace('[€,]', '', regex=True).astype(float)
    validation_data[column] = validation_data[column].replace('[€,]', '', regex=True).astype(float)

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

# Define the SARIMAX models for each location's revenue
for column in location_columns:
    y_train = train_data[column]
    X_train = train_data[['time_trend']]  # You can include other exogenous variables here if available
    y_valid = validation_data[column]
    X_valid = validation_data[['time_trend']]  # Exogenous variables for validation
    
    # Fit the SARIMAX model
    sarimax_models[column] = sm.tsa.SARIMAX(endog=y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarimax_results[column] = sarimax_models[column].fit()
    
    # Forecast on the validation set
    forecasts[column] = sarimax_results[column].forecast(steps=len(y_valid), exog=X_valid)
    
    # Calculate RMSE and MAE for each location
    rmse_values[column] = int(np.sqrt(mean_squared_error(y_valid, forecasts[column])))
    mae_values[column] = int(mean_absolute_error(y_valid, forecasts[column]))
    
    # Print the forecasted values
    print(f"Forecasted Revenue for {column}:")
    print(forecasts[column].apply(lambda x: f'{int(x):,d}'))

# Combine the forecasts into total revenue forecast (sum of all locations)
forecast_total = pd.DataFrame(forecasts).sum(axis=1)

# Print total forecasted revenue
print("Total Forecasted Revenue:")
print(forecast_total.apply(lambda x: f'{int(x):,d}'))

# Print RMSE and MAE for each location and total revenue
for column in location_columns:
    print(f"RMSE for {column}: {rmse_values[column]:,d}")
    print(f"MAE for {column}: {mae_values[column]:,d}")

# Total RMSE and MAE for all locations combined
rmse_total = int(np.sqrt(mean_squared_error(validation_data['Revenue Total'], forecast_total)))
mae_total = int(mean_absolute_error(validation_data['Revenue Total'], forecast_total))

print(f"RMSE for Total Revenue: {rmse_total:,d}")
print(f"MAE for Total Revenue: {mae_total:,d}")

# Combine the actual revenue from training and validation datasets for plotting
actual_total_train = train_data[location_columns].sum(axis=1)
actual_total_valid = validation_data[location_columns].sum(axis=1)

# Plot the actual vs forecast for total revenue
plt.figure(figsize=(10,6))
plt.plot(actual_total_train.index, actual_total_train, label='Actual Total Revenue (Training)', color='blue')
plt.plot(actual_total_valid.index, actual_total_valid, label='Actual Total Revenue (Validation)', color='blue', linestyle='--')

# Only plot forecasted revenue for the validation period
plt.plot(validation_data.index, forecast_total, label='Forecasted Total Revenue (Validation)', linestyle='--', color='red')

plt.title('Actual vs Forecasted Total Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue (€)')
plt.legend()
plt.show()

# Print monthly forecasting errors in integer format
monthly_errors = pd.DataFrame({
    'Actual Total Revenue': validation_data['Revenue Total'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Total Revenue': forecast_total.apply(lambda x: f'{int(x):,d}')
})

print("\nMonthly Forecasting Errors:")
print(monthly_errors)
monthly_errors.to_csv('errors.csv')