import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from forecast_utils import ForecastUtils

# Revenue forecasting model utilizin SARIMAX and wrapping 4 services
# - Initializion
# - Training
# - Forecasting
# - Validation
#
# The model uses a unified dataset read in initialization and functions only get dates as parameters
class RevenueForecastRunrate:

    def __init__(self, use_trend = False):
        self.data = ForecastUtils.load_rate_normalized_data()
        self.use_trend = use_trend


    def train_model(self, train_start, train_end):

        def calculate_linear_trend(series):
            
            trend_length = len(series)
            if len(series) >= 12:
                trend_lenght = 12
            
            X = np.arange(len(series[-trend_lenght:])).reshape(-1, 1)
            y = series[-trend_lenght:].values
            model = LinearRegression()
            model.fit(X, y)
            return model.coef_[0]
    
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
        self.train_data.loc[:, 'Normalized Revenue FIN'] = self.train_data['Revenue FIN'] / (self.train_data['Month'].map(self.cyclicity_FIN) * self.train_data['D/M FIN'])
        self.train_data.loc[:, 'Normalized Revenue IND'] = self.train_data['Revenue IND'] / (self.train_data['Month'].map(self.cyclicity_IND) * self.train_data['D/M IND'])
        self.train_data.loc[:, 'Normalized Revenue NS'] = self.train_data['Revenue NS'] / (self.train_data['Month'].map(self.cyclicity_NS) * self.train_data['D/M NS'])

        # Calculate trend from the last six months of the training data if trend is used.
        # Using trend is defined in the initialization of the model. Default = False.
        if(self.use_trend):
            self.trend_FIN = calculate_linear_trend(self.train_data['Normalized Revenue FIN'])
            self.trend_IND = calculate_linear_trend(self.train_data['Normalized Revenue IND'])
            self.trend_NS = calculate_linear_trend(self.train_data['Normalized Revenue NS'])

        # Use the last five months of training data to calculate weighted run rate
        self.run_rate_period = self.train_data.tail(5)


    def forecast(self, forecast_start, forecast_end):

        forecast_data = ForecastUtils.split_data(self.data, forecast_start, forecast_end)        

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
        forecast_run_rate_FIN = weighted_run_rate(self.run_rate_period['Normalized Revenue FIN'])
        forecast_run_rate_IND = weighted_run_rate(self.run_rate_period['Normalized Revenue IND'])
        forecast_run_rate_NS = weighted_run_rate(self.run_rate_period['Normalized Revenue NS'])

        forecast_months = forecast_data['Month']

        # Forecast for each revenue stream using the weighted run rates and cyclicity adjusted by working days
        forecast_FIN = None
        forecast_IND = None
        forecast_NS = None

        if(self.use_trend):
            forecast_FIN = [self.cyclicity_FIN[month] * (forecast_run_rate_FIN + self.trend_FIN * (i+1)) * forecast_data.loc[forecast_data['Month'] == month, 'D/M FIN'].values[0] for i, month in enumerate(forecast_months)]
            forecast_IND = [self.cyclicity_IND[month] * (forecast_run_rate_FIN + self.trend_IND * (i+1)) * forecast_data.loc[forecast_data['Month'] == month, 'D/M IND'].values[0] for i, month in enumerate(forecast_months)]
            forecast_NS = [self.cyclicity_NS[month] * (forecast_run_rate_FIN + self.trend_NS * (i+1)) * forecast_data.loc[forecast_data['Month'] == month, 'D/M NS'].values[0] for i, month in enumerate(forecast_months)]
        else:
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

    # Instantiate two services: one to forecast without trend, one with trend
    service_without_trend = RevenueForecastRunrate() 
    service_with_trend = RevenueForecastRunrate(use_trend=True) 

    # Train the model (it's internal submodels)
    service_without_trend.train_model('2021-10-01', '2024-03-31') 
    service_with_trend.train_model('2021-10-01', '2024-03-31') 

    # Forecasting
    forecast_fin, forecast_ind, forecast_ns, forecast_total = service_without_trend.forecast('2024-04-01', '2024-09-30')
    forecast_with_trend_fin, forecast_with_trend_ind, forecast_with_trend_ns, forecast_with_trend_total = service_with_trend.forecast('2024-04-01', '2024-09-30')

    # Validating
    service_without_trend.validate(forecast_fin, forecast_ind, forecast_ns, '2024-04-01', '2024-09-30')    
    service_with_trend.validate(forecast_with_trend_fin, forecast_with_trend_ind, forecast_with_trend_ns, '2024-04-01', '2024-09-30')    

