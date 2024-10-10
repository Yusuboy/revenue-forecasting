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

    return df

# Split the data to training and validation datasets
def split_data(df, train_start, train_end, validatation_start, validatation_end, forecast_start, forecast_end):
    train_data = df[(df.index >= train_start) & (df.index <= train_end)]
    validation_data = df[(df.index >= validatation_start) & (df.index <= validatation_end)]
    forecast_data = df[(df.index >= forecast_start) & (df.index <= forecast_end)]

    print(f'training data {train_data.shape}')
    print(train_data)
    print(f'validation data {validation_data.shape}')
    print(validation_data)
    print(f'forecast data {forecast_data.shape}')
    print(forecast_data)

    return train_data, validation_data, forecast_data

def create_models(train_data):

    # Define SARIMAX model parameters
    order = (1, 1, 1)  # (p, d, q) parameters for ARIMA
    seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s) for seasonal component (annual seasonality)

    # Explanatory variables for each model
    exog_FIN = train_data[['D/M FIN', 'Easter']]
    exog_IND = train_data[['D/M IND']]
    exog_NS = train_data[['D/M NS', 'Easter']]

    # SARIMAX model for Revenue FIN
    model_FIN = sm.tsa.statespace.SARIMAX(train_data['Revenue FIN'],
                                        exog=exog_FIN,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        trend='ct')  # 'ct' includes both constant and linear trend
    results_FIN = model_FIN.fit()

    # SARIMAX model for Revenue IND
    model_IND = sm.tsa.statespace.SARIMAX(train_data['Revenue IND'],
                                        exog=exog_IND,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        trend='ct')
    results_IND = model_IND.fit()

    # SARIMAX model for Revenue NS
    model_NS = sm.tsa.statespace.SARIMAX(train_data['Revenue NS'],
                                        exog=exog_NS,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        trend='ct')
    results_NS = model_NS.fit()

    return results_FIN, results_IND, results_NS

data = load_data()

train_start = '2021-10-01'
train_end = '2024-03-31'
validation_start = '2024-04-01'
validation_end = '2024-09-30'
forecast_start = '2024-10-01'
forecast_end = '2025-09-30'

train_data, validation_data, forecast_data = split_data(data, train_start, train_end, validation_start, validation_end, forecast_start, forecast_end)

results_FIN, results_IND, results_NS = create_models(train_data)

# Prepare explanatory variables for validation data
exog_FIN_val = validation_data[['D/M FIN', 'Easter']]
exog_IND_val = validation_data[['D/M IND']]
exog_NS_val = validation_data[['D/M NS', 'Easter']]

exog_FIN_fc = forecast_data[['D/M FIN', 'Easter']]
exog_IND_fc = forecast_data[['D/M IND']]
exog_NS_fc = forecast_data[['D/M NS', 'Easter']]

# Forecasting for the validation period (April 2024 to September 2024)
validation_FIN = results_FIN.predict(start=validation_data.index[0], end=validation_data.index[-1], exog=exog_FIN_val)
validation_IND = results_IND.predict(start=validation_data.index[0], end=validation_data.index[-1], exog=exog_IND_val)
validation_NS = results_NS.predict(start=validation_data.index[0], end=validation_data.index[-1], exog=exog_NS_val)

print(exog_FIN_fc.shape)
print(exog_FIN_fc)

# Forecasting the additional forecast period 10-12/2024 where there is no validation data
#forecast_FIN = results_FIN.predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_FIN_fc)
#forecast_IND = results_IND.predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_IND_fc)
#forecast_NS = results_NS.predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_NS_fc)

# Combine the forecasted values for total revenue
validation_total = validation_FIN + validation_IND + validation_NS
#forecast_total = forecast_FIN + forecast_FIN + forecast_NS

# Actual total revenue from the validation set
actual_validation_total = validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS']

# Evaluate the model using RMSE and MAE for the total revenue
rmse_total = np.sqrt(mean_squared_error(actual_validation_total, validation_total))
mae_total = mean_absolute_error(actual_validation_total, validation_total)

# Print results
#print(f"RMSE for Total Revenue (Validation Period): {rmse_total}")
#print(f"MAE for Total Revenue (Validation Period): {mae_total}")

# Print individual model summaries
#print("\nSARIMAX Model Summary for Revenue FIN:")
#print(results_FIN.summary())

#print("\nSARIMAX Model Summary for Revenue IND:")
#print(results_IND.summary())

#print("\nSARIMAX Model Summary for Revenue NS:")
#print(results_NS.summary())

# Create the DataFrame to compare actual and forecasted revenues
monthly_errors = pd.DataFrame({
    'Actual Revenue FIN': validation_data['Revenue FIN'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue FIN': pd.Series(validation_FIN, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue IND': validation_data['Revenue IND'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue IND': pd.Series(validation_IND, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Revenue NS': validation_data['Revenue NS'].apply(lambda x: f'{int(x):,d}'),
    'Forecasted Revenue NS': pd.Series(validation_NS, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
    'Actual Total Revenue': actual_validation_total.apply(lambda x: f'{int(x):,d}'),
    'Forecasted Total Revenue': pd.Series(validation_total, index=validation_data.index).apply(lambda x: f'{int(x):,d}')
})

monthly_errors.to_csv('errors_sarimax.csv')

# Add Error% FIN, Error% IND, and Error% NS columns
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

# Adding Total Revenue Error %
monthly_errors['Error% Total'] = monthly_errors.apply(
    lambda row: None if row['Actual Total Revenue'] is None else 
    f"{((float(row['Actual Total Revenue'].replace(',', '')) - float(row['Forecasted Total Revenue'].replace(',', ''))) / float(row['Actual Total Revenue'].replace(',', '')) * 100):.2f}%", axis=1
)

# Display the final DataFrame
print("\nMonthly Forecasting Errors (Actual vs Forecasted with Errors):")
print(monthly_errors)

# Create a DataFrame for the monthly forecast including Year, Month, D/M fields and forecasted revenues
#forecast_per_month = pd.DataFrame({
#    'Year': forecast_data['Year'],
#    'Month': forecast_data['Month'],
#    'D/M FIN': forecast_data['D/M FIN'],
#    'D/M IND': forecast_data['D/M IND'],
#    'D/M NS': forecast_data['D/M NS'],
#    'Forecast FIN': pd.Series(forecast_FIN, index=forecast_data.index).apply(lambda x: f'{int(x):,d}'),
#    'Forecast IND': pd.Series(forecast_IND, index=forecast_data.index).apply(lambda x: f'{int(x):,d}'),
#    'Forecast NS': pd.Series(forecast_NS, index=forecast_data.index).apply(lambda x: f'{int(x):,d}')
#})

# Plotting the actual vs forecasted revenue for the validation period and the training period
plt.figure(figsize=(10, 6))
#print("\nMonthly Forecasting for 12 months:")
#print(forecast_per_month)

# Plot training period (actual data)
plt.plot(train_data.index, train_data['Revenue FIN'] + train_data['Revenue IND'] + train_data['Revenue NS'], label='Training Total Revenue', color='blue')

# Plot validation period (actual vs forecasted)
plt.plot(validation_data.index, actual_validation_total, label='Actual Total Revenue (Validation)', color='green')
plt.plot(validation_data.index, validation_total, label='Forecasted Total Revenue (Validation)', color='red', linestyle='--')

# Labels and legend
plt.title('Actual vs Forecasted Total Revenue (Training and Validation)')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()