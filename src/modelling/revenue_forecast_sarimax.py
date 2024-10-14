import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecast_utils import ForecastUtils
from pmdarima import auto_arima

# Revenue forecasting model utilizin SARIMAX and wrapping 4 services
# - Initializion
# - Training
# - Forecasting
# - Validation
#
# The model uses a unified dataset read in initialization and functions only get dates as parameters
class RevenueForecastSarimax:

    # Initialize the model. 
    # Exoc variables are using obvious defaults but can be overriden by user if needed.
    def __init__(self, exoc_variables_fin=['D/M FIN'], exoc_variables_ind = ['D/M IND'], exoc_variables_ns = ['D/M NS']):

        self.data = ForecastUtils.load_rate_normalized_data()
        self.models = {}
        self.exoc_variables_fin = exoc_variables_fin
        self.exoc_variables_ind = exoc_variables_ind
        self.exoc_variables_ns = exoc_variables_ns

    # Train the sarimax models used in forecasting. Trained models are stored inside the object for further use.
    def train_model(self, train_start, train_end):
        
        self.train_data = ForecastUtils.split_data(self.data, train_start, train_end)

        # Fit auto_arima to determine best model
        #model = auto_arima(self.train_data['Revenue FIN'],
        #                exogenous=self.train_data[self.exoc_variables_fin],
        #                seasonal=True,
        #                m=12,  # s=12 for yearly seasonality
        #                D=1,   # Manually set seasonal differencing
        #                trace=True,
        #                stepwise=True)
        #model.summary()

        # Define SARIMAX model parameters
        # FIN Best model:  ARIMA(1,1,0)(0,1,0)[12]
        order = (1, 1, 1)  # (p, d, q) parameters for ARIMA
        seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s) for seasonal component (annual seasonality)

        # SARIMAX model for Revenue FIN
        model_FIN = sm.tsa.statespace.SARIMAX(self.train_data['Revenue FIN'],
                                            exog=self.train_data[self.exoc_variables_fin],
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            trend='ct')  # 'ct' includes both constant and linear trend
        self.models['FIN'] = model_FIN.fit()

        # SARIMAX model for Revenue IND
        model_IND = sm.tsa.statespace.SARIMAX(self.train_data['Revenue IND'],
                                            exog=self.train_data[self.exoc_variables_ind],
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            trend='ct')
        self.models['IND'] = model_IND.fit()

        # SARIMAX model for Revenue NS
        model_NS = sm.tsa.statespace.SARIMAX(self.train_data['Revenue NS'],
                                            exog=self.train_data[self.exoc_variables_ns],
                                            order=order,
                                            seasonal_order=seasonal_order,
                                            trend='ct')
        self.models['NS'] = model_NS.fit()

    # Forecast monthas between start and end dates given as parameter.
    # Because of the Sarimax seasonality, the shortes forecast period data is 7 months. 
    # If shorter forecast is needed, you need to have the exoc data for 7 months available and thake the shorter period out of the result.
    def forecast(self, forecast_start, forecast_end):

        forecast_data = ForecastUtils.split_data(self.data, forecast_start, forecast_end)

        # Extract exogenous variables for the forecast period
        exog_fin = forecast_data[self.exoc_variables_fin]
        exog_ind = forecast_data[self.exoc_variables_ind]
        exog_ns = forecast_data[self.exoc_variables_ns]

        # Perform forecasting using the trained models
        forecast_fin = self.models['FIN'].predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_fin)
        forecast_ind = self.models['IND'].predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_ind)
        forecast_ns = self.models['NS'].predict(start=forecast_data.index[0], end=forecast_data.index[-1], exog=exog_ns)

        # Return combined forecast
        forecast_total = forecast_fin + forecast_ind + forecast_ns
        return forecast_fin, forecast_ind, forecast_ns, forecast_total

    # Utility function to validate the model
    def validate(self, forecast_fin, forecast_ind, forecast_ns, validation_start, validation_end, save_errors=False, plot_errors=False):
        validation_data = ForecastUtils.split_data(self.data, validation_start, validation_end)
        result = ForecastUtils.validation_results(validation_data, forecast_fin, forecast_ind, forecast_ns, save_errors=save_errors, plot_errors=plot_errors, model_name='sarimax', train_data=self.train_data)
        return result
    
# Usage example / dummy test
if __name__ == "__main__":  
    
    # Instantiate the service
    service = RevenueForecastSarimax() 

    # Train the model (it's internal submodels)
    service.train_model('2021-10-01', '2024-03-31') 

    # Forecasting
    forecast_fin, forecast_ind, forecast_ns, forecast_total = service.forecast('2024-04-01', '2024-09-30')

    # Validating
    service.validate(forecast_fin, forecast_ind, forecast_ns, '2024-04-01', '2024-09-30')

