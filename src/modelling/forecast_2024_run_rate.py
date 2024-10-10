import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
def load_data():
    df = pd.read_csv('data.csv')

    # Create DateTime indices
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df.set_index('Date', inplace=True)
  
    # Ensure that decimal fields are correct and numeric
    df['Revenue FIN'] = df['Revenue FIN'].astype(float)
    df['Revenue IND'] = df['Revenue IND'].astype(float)
    df['Revenue NS'] = df['Revenue NS'].astype(float)
    df['Hours 10* FI'] = df['Hours 10* FI'].fillna(0).astype(float)
    df['Hours 20* FI'] = df['Hours 20* FI'].fillna(0).astype(float)
    df['Hours 30* FI'] = df['Hours 30*FI'].fillna(0).astype(float)
    df['Hours 10* IN'] = df['Hours 10* IN'].fillna(0).astype(float)
    df['Hours 20* IN'] = df['Hours 20* IN'].fillna(0).astype(float)
    df['Hours 30* IN'] = df['Hours 30*IN'].fillna(0).astype(float)
    df['Hours 10* NS'] = df['Hours 10* NS'].fillna(0).astype(float)
    df['Hours 20* NS'] = df['Hours 20* NS'].fillna(0).astype(float)
    df['Hours 30* NS'] = df['Hours 30*NS'].fillna(0).astype(float)

    return df

# Split the data to training and validation datasets
def split_data(df, train_start, train_end, validatation_start, validatation_end, forecast_start, forecast_end):
    train_data = df[(df.index >= train_start) & (df.index <= train_end)]
    validation_data = df[(df.index >= validatation_start) & (df.index <= validatation_end)]
    forecast_data = df[(df.index >= forecast_start) & (df.index <= forecast_end)]

    print(forecast_data)

    return train_data, validation_data, forecast_data


data = load_data()
train_start = '2022-01-01'
train_end = '2024-03-31'
validation_start = '2024-04-01'
validation_end = '2024-09-30'
forecast_start = '2024-10-01'
forecast_end = '2024-12-31'
train_data, validation_data, forecast_data = split_data(data, train_start, train_end, validation_start, validation_end, forecast_start, forecast_end)

# Define the cyclicity based on full years (2022 and 2023) for forecasting
cyclic_years = train_data[(train_data['Year'] == 2022) | (train_data['Year'] == 2023)]

# Calculate cyclicity for each revenue category, taking into account the working days
cyclicity_FIN = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue FIN'] / x['D/M FIN']).mean())
cyclicity_IND = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue IND'] / x['D/M IND']).mean())
cyclicity_NS = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue NS'] / x['D/M NS']).mean())

# Normalize the revenues in both train and validation sets using the cyclicity multipliers and the number of working days
train_data['Normalized Revenue FIN'] = train_data['Revenue FIN'] / (train_data['Month'].map(cyclicity_FIN) * train_data['D/M FIN'])
train_data['Normalized Revenue IND'] = train_data['Revenue IND'] / (train_data['Month'].map(cyclicity_IND) * train_data['D/M IND'])
train_data['Normalized Revenue NS'] = train_data['Revenue NS'] / (train_data['Month'].map(cyclicity_NS) * train_data['D/M NS'])

validation_data['Normalized Revenue FIN'] = validation_data['Revenue FIN'] / (validation_data['Month'].map(cyclicity_FIN) * validation_data['D/M FIN'])
validation_data['Normalized Revenue IND'] = validation_data['Revenue IND'] / (validation_data['Month'].map(cyclicity_IND) * validation_data['D/M IND'])
validation_data['Normalized Revenue NS'] = validation_data['Revenue NS'] / (validation_data['Month'].map(cyclicity_NS) * validation_data['D/M NS'])

# Use the last five months to calculate weighted run rate
# Use last months of training period for model validation and last months of validation period for actual 2024 forecast
validation_run_rate_period = train_data.tail(5)
forecast_run_rate_period = validation_data.tail(5)

# Define the weights for the last five months
weights = np.array([5, 4, 3, 2, 1])

# Weighted average run rate for each revenue stream
def weighted_run_rate(series):
    if len(series) >= 5:
        return np.average(series[-5:], weights=weights)
    else:
        truncated_weights = weights[-len(series):]
        return np.average(series, weights=truncated_weights)

# Calculate the weighted run rates using the adjusted function
validation_run_rate_FIN = weighted_run_rate(validation_run_rate_period['Normalized Revenue FIN'])
validation_run_rate_IND = weighted_run_rate(validation_run_rate_period['Normalized Revenue IND'])
validation_run_rate_NS = weighted_run_rate(validation_run_rate_period['Normalized Revenue NS'])

forecast_run_rate_FIN = weighted_run_rate(forecast_run_rate_period['Normalized Revenue FIN'])
forecast_run_rate_IND = weighted_run_rate(forecast_run_rate_period['Normalized Revenue IND'])
forecast_run_rate_NS = weighted_run_rate(forecast_run_rate_period['Normalized Revenue NS'])

# Extend the validation period to include October, November, and December 2024
validation_months = validation_data['Month']
forecast_months = forecast_data['Month']

# Forecast for each revenue stream using the weighted run rates and cyclicity adjusted by working days
validation_forecast_FIN = [cyclicity_FIN[month] * validation_run_rate_FIN * validation_data.loc[validation_data['Month'] == month, 'D/M FIN'].values[0] for month in validation_months]
validation_forecast_IND = [cyclicity_IND[month] * validation_run_rate_IND * validation_data.loc[validation_data['Month'] == month, 'D/M IND'].values[0] for month in validation_months]
validation_forecast_NS = [cyclicity_NS[month] * validation_run_rate_NS * validation_data.loc[validation_data['Month'] == month, 'D/M NS'].values[0] for month in validation_months]

forecast_forecast_FIN = [cyclicity_FIN[month] * forecast_run_rate_FIN * forecast_data.loc[forecast_data['Month'] == month, 'D/M FIN'].values[0] for month in forecast_months]
forecast_forecast_IND = [cyclicity_IND[month] * forecast_run_rate_IND * forecast_data.loc[forecast_data['Month'] == month, 'D/M IND'].values[0] for month in forecast_months]
forecast_forecast_NS = [cyclicity_NS[month] * forecast_run_rate_NS * forecast_data.loc[forecast_data['Month'] == month, 'D/M NS'].values[0] for month in forecast_months]

# Convert to numpy arrays for calculation
validation_forecast_FIN = np.array(validation_forecast_FIN)
validation_forecast_IND = np.array(validation_forecast_IND)
validation_forecast_NS = np.array(validation_forecast_NS)

forecast_forecast_FIN = np.array(forecast_forecast_FIN)
forecast_forecast_IND = np.array(forecast_forecast_IND)
forecast_forecast_NS = np.array(forecast_forecast_NS)

# Calculate total forecasted revenue
validation_forecast_total = validation_forecast_FIN + validation_forecast_IND + validation_forecast_NS

# Extract actual values for validation
y_valid_FIN = validation_data['Revenue FIN']
y_valid_IND = validation_data['Revenue IND']
y_valid_NS = validation_data['Revenue NS']

# Calculate the actual total revenue for the validation period
actual_total = validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS']

# Calculate RMSE and MAE for the total revenue
rmse_total = np.sqrt(mean_squared_error(actual_total, validation_forecast_total))
mae_total = mean_absolute_error(actual_total, validation_forecast_total)

# Output RMSE and MAE
print(f"RMSE for Total Revenue: {rmse_total}")
print(f"MAE for Total Revenue: {mae_total}")

forecast_index = forecast_data.index
monthly_errors = pd.DataFrame({
    'Actual Revenue FIN': y_valid_FIN.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue FIN': pd.Series(validation_forecast_FIN, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue IND': y_valid_IND.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue IND': pd.Series(validation_forecast_IND, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue NS': y_valid_NS.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue NS': pd.Series(validation_forecast_NS, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Total Revenue': actual_total.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Total Revenue': pd.Series(validation_forecast_total, index=validation_data.index).apply(lambda x: f'{int(x):,d}')
})

# Add Error% FIN and Error% IND columns, handling cases where Actual Revenue is None
monthly_errors['Error% FIN'] = monthly_errors.apply(
    lambda row: None if row['Actual Revenue FIN'] is None else 
    f"{((float(row['Actual Revenue FIN'].replace(',', '')) - float(row['Forecasted Revenue FIN'].replace(',', ''))) / float(row['Actual Revenue FIN'].replace(',', '')) * 100):.2f}%", axis=1
)

monthly_errors['Error% IND'] = monthly_errors.apply(
    lambda row: None if row['Actual Revenue IND'] is None else 
    f"{((float(row['Actual Revenue IND'].replace(',', '')) - float(row['Forecasted Revenue IND'].replace(',', ''))) / float(row['Actual Revenue IND'].replace(',', '')) * 100):.2f}%", axis=1
)

monthly_errors['Error% NS'] = monthly_errors.apply(
    lambda row: None if row['Actual Revenue NS'] is None else 
    f"{((float(row['Actual Revenue NS'].replace(',', '')) - float(row['Forecasted Revenue NS'].replace(',', ''))) / float(row['Actual Revenue NS'].replace(',', '')) * 100):.2f}%", axis=1
)

monthly_errors.to_csv('errors_runrate.csv')

# Combine validation and forecast periods
validation_errors = pd.DataFrame({
    'Actual Revenue FIN': validation_data['Revenue FIN'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue FIN': pd.Series(validation_forecast_FIN, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue IND': validation_data['Revenue IND'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue IND': pd.Series(validation_forecast_IND, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue NS': validation_data['Revenue NS'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue NS': pd.Series(validation_forecast_NS, index=validation_data.index).apply(lambda x: f'{int(x):,d}')
})


# Print the updated table with forecasted and actual revenues side by side
print("\nValidation Period Errors (Actual vs Forecasted):")
print(monthly_errors)

# Create a DataFrame for the monthly forecast including Year, Month, D/M fields and forecasted revenues
forecast_per_month = pd.DataFrame({
    'Year': forecast_data['Year'],
    'Month': forecast_data['Month'],
    'D/M FIN': forecast_data['D/M FIN'],
    'D/M IND': forecast_data['D/M IND'],
    'D/M NS': forecast_data['D/M NS'],
    'Forecast FIN': pd.Series(forecast_forecast_FIN, index=forecast_data.index).apply(lambda x: f'{int(x):,d}'),
    'Forecast IND': pd.Series(forecast_forecast_IND, index=forecast_data.index).apply(lambda x: f'{int(x):,d}'),
    'Forecast NS': pd.Series(forecast_forecast_NS, index=forecast_data.index).apply(lambda x: f'{int(x):,d}')
})

print("\nForecast per Month:")
print(forecast_per_month)

# Plot forecasted vs actual Revenue IND and Revenue FIN, including the extended forecast period
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(train_data.index, train_data['Revenue FIN'], label='Actual Revenue FIN (Training)', color='blue', linestyle='--')
plt.plot(train_data.index, train_data['Revenue IND'], label='Actual Revenue IND (Training)', color='green', linestyle='--')

# Plot validation data
plt.plot(validation_data.index, validation_data['Revenue FIN'], label='Actual Revenue FIN (Validation)', color='blue')
plt.plot(validation_data.index, validation_forecast_FIN, label='Forecasted Revenue FIN (Validation)', linestyle='--', color='red')
plt.plot(validation_data.index, validation_data['Revenue IND'], label='Actual Revenue IND (Validation)', color='green')
plt.plot(validation_data.index, validation_forecast_IND, label='Forecasted Revenue IND (Validation)', linestyle='--', color='orange')

# Plot forecast period data
plt.plot(forecast_data.index, forecast_forecast_FIN, label='Forecasted Revenue FIN (Forecast)', linestyle='--', color='red')
plt.plot(forecast_data.index, forecast_forecast_IND, label='Forecasted Revenue IND (Forecast)', linestyle='--', color='orange')

plt.title('Forecasted vs Actual Revenue for IND and FIN (Including Extended Forecast Period)')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.show()