import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from forecast_utils import ForecastUtils

# Revenue forecasting model utilizin SARIMAX and wrapping 4 services
# - Initializion
# - Training
# - Forecasting
# - Validation
#
# The model uses a unified dataset read in initialization and functions only get dates as parameters
class RevenueForecastRunrate:

    def __init__(self):
        self.data = ForecastUtils.load_rate_normalized_data()

    def train_model(self, train_start, train_end):

        self.train_data = ForecastUtils.split_data(self.data, train_start, train_end)

        # Define the cyclicity based on full years found in the train data
        first_year = self.train_data.iloc[0]['Year'] if (self.train_data.iloc[0]['Month']==1) else self.train_data.iloc[0]['Year']+1
        last_year = self.train_data.iloc[-1]['Year'] if (self.train_data.iloc[-1]['Month']==12) else self.train_data.iloc[0]['Year']-1
        cyclic_years = self.train_data[(self.train_data['Year'] < first_year) | (self.train_data['Year'] > last_year)]

        # Calculate cyclicity for each revenue category, taking into account the working days
        self.cyclicity_FIN = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue FIN'] / x['D/M FIN']).mean())
        self.cyclicity_IND = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue IND'] / x['D/M IND']).mean())
        self.cyclicity_NS = cyclic_years.groupby('Month').apply(lambda x: (x['Revenue NS'] / x['D/M NS']).mean())      

        # Normalize the revenues in train data
        self.train_data['Normalized Revenue FIN'] = self.train_data['Revenue FIN'] / (self.train_data['Month'].map(self.cyclicity_FIN) * self.train_data['D/M FIN'])
        self.train_data['Normalized Revenue IND'] = self.train_data['Revenue IND'] / (self.train_data['Month'].map(self.cyclicity_IND) * self.train_data['D/M IND'])
        self.train_data['Normalized Revenue NS'] = self.train_data['Revenue NS'] / (self.train_data['Month'].map(self.cyclicity_NS) * self.train_data['D/M NS'])  

    def forecast(self, forecast_start, forecast_end):

        forecast_data = ForecastUtils.split_data(self.data, forecast_start, forecast_end)        

        # Normalize the revenues in forecast data
        # forecast_data['Normalized Revenue FIN'] = forecast_data['Revenue FIN'] / (forecast_data['Month'].map(self.cyclicity_FIN) * forecast_data['D/M FIN'])
        # forecast_data['Normalized Revenue IND'] = forecast_data['Revenue IND'] / (forecast_data['Month'].map(self.cyclicity_IND) * forecast_data['D/M IND'])
        # forecast_data['Normalized Revenue NS'] = forecast_data['Revenue NS'] / (forecast_data['Month'].map(self.cyclicity_NS) * forecast_data['D/M NS'])

        # Use the last five months of training data to calculate weighted run rate
        run_rate_period = self.train_data.tail(5)

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
        forecast_run_rate_FIN = weighted_run_rate(run_rate_period['Normalized Revenue FIN'])
        forecast_run_rate_IND = weighted_run_rate(run_rate_period['Normalized Revenue IND'])
        forecast_run_rate_NS = weighted_run_rate(run_rate_period['Normalized Revenue NS'])

        forecast_months = forecast_data['Month']

        # Forecast for each revenue stream using the weighted run rates and cyclicity adjusted by working days
        forecast_FIN = [self.cyclicity_FIN[month] * forecast_run_rate_FIN * forecast_data.loc[forecast_data['Month'] == month, 'D/M FIN'].values[0] for month in forecast_months]
        forecast_IND = [self.cyclicity_IND[month] * forecast_run_rate_IND * forecast_data.loc[forecast_data['Month'] == month, 'D/M IND'].values[0] for month in forecast_months]
        forecast_NS = [self.cyclicity_NS[month] * forecast_run_rate_NS * forecast_data.loc[forecast_data['Month'] == month, 'D/M NS'].values[0] for month in forecast_months]

        # Convert to numpy arrays for calculation
        forecast_FIN = np.array(forecast_FIN)
        forecast_IND = np.array(forecast_IND)
        forecast_NS = np.array(forecast_NS)
        forecast_total = forecast_FIN + forecast_IND + forecast_NS

        return forecast_FIN, forecast_IND, forecast_NS, forecast_total

    # Utility function to validate the model
    def validate(self, forecast_fin, forecast_ind, forecast_ns, validation_start, validation_end):
        validation_data = ForecastUtils.split_data(self.data, validation_start, validation_end)
        ForecastUtils.validation_results(validation_data, forecast_fin, forecast_ind, forecast_ns, save_errors=False, plot_errors=True, train_data=self.train_data)

# Usage example / dummy test
if __name__ == "__main__":  

    # Instantiate the service
    service = RevenueForecastRunrate() 

    # Train the model (it's internal submodels)
    service.train_model('2021-10-01', '2024-03-31') 

    # Forecasting
    forecast_fin, forecast_ind, forecast_ns, forecast_total = service.forecast('2024-04-01', '2024-09-30')

    # Validating
    service.validate(forecast_fin, forecast_ind, forecast_ns, '2024-04-01', '2024-09-30')    