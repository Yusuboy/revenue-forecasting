import pandas as pd
import numpy as np
import statsmodels.api as sm
from forecast_utils import ForecastUtils
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

# Revenue forecasting model utilizin multicaptive .
# 
# Four services provided:
# - Initializion
# - Training
# - Forecasting
# - Validation
#
# The model uses a unified dataset read in initialization and functions only get dates as parameters
class RevenueForecastMulticaptive2:

    # Initialize the model. 
    def __init__(self):

        self.data = ForecastUtils.load_rate_normalized_data()
        self.data = self.data.rename(columns={'Unnamed: 0': 'Month_index'})

        # Select relevant columns and drop na
        self.data_models = self.data[['Month_index', 'D/M FIN', 'Revenue FIN', 'Revenue IND', 'Revenue NS']]
        self.data_models = self.data_models.dropna(subset = ['Revenue FIN'])
        self.data_models = self.data_models.iloc[:-1]        
        
        # This needs to be commented out to allow working dynamically with parametrized dates.
        # In later phases, we just need to ensure that data is handled properly based on parametrized dates rather than location in dataframe.
        #data_models = data_models.iloc[:-1] 
    
    # Initialize the models using data in the given training period.
    def train_model(self, train_start, train_end):

        # Self.train_data is only to allow using the common poltting utility function later.
        # The actual training data is in variable 'training_data'.
        self.train_data = ForecastUtils.split_data(self.data, train_start, train_end)  

        # This is the actual used training data
        training_data = ForecastUtils.split_data(self.data_models, train_start, train_end)
        #training_data = self.data_models[:-6] - if this would be used, 10/24 should be manually dropped

        # Include working days in the training data
        training_data['Working_days_FIN'] = self.train_data['D/M FIN']
        training_data['Working_days_IND'] = self.train_data['D/M IND']
        training_data['Working_days_NS'] = self.train_data['D/M NS']

        # Normalize revenue by working days for each revenue type
        training_data['Revenue_FIN_normalized'] = training_data['Revenue FIN'] / training_data['Working_days_FIN']
        training_data['Revenue_IND_normalized'] = training_data['Revenue IND'] / training_data['Working_days_IND']
        training_data['Revenue_NS_normalized'] = training_data['Revenue NS'] / training_data['Working_days_NS']

        # Get the columns for each model in a separate dataframe
        #data_FIN = training_data[['Month_index', 'Revenue FIN']]
        #data_IND = training_data[['Month_index', 'Revenue IND']]
        #data_NS = training_data[['Month_index', 'Revenue NS']]
        #data_FIN = data_FIN.set_index('Month_index')
        #data_IND = data_IND.set_index('Month_index')
        #data_NS = data_NS.set_index('Month_index')

        # Set index for each revenue type
        data_FIN = training_data[['Month_index', 'Revenue_FIN_normalized']].set_index('Month_index')
        data_IND = training_data[['Month_index', 'Revenue_IND_normalized']].set_index('Month_index')
        data_NS = training_data[['Month_index', 'Revenue_NS_normalized']].set_index('Month_index')

        data_FIN_train = data_FIN
        data_IND_train = data_IND
        data_NS_train = data_NS

        # Create models
        #self.multiplicative_model_FIN = seasonal_decompose(data_FIN_train['Revenue FIN'], model='multiplicative', period=12)
        #self.multiplicative_model_IND = seasonal_decompose(data_IND_train['Revenue IND'], model='multiplicative', period=12)
        #self.multiplicative_model_NS = seasonal_decompose(data_NS_train['Revenue NS'], model='multiplicative', period=12)

        # Seasonal decomposition on normalized revenue
        self.multiplicative_model_FIN = seasonal_decompose(data_FIN['Revenue_FIN_normalized'], model='multiplicative', period=12)
        self.multiplicative_model_IND = seasonal_decompose(data_IND['Revenue_IND_normalized'], model='multiplicative', period=12)
        self.multiplicative_model_NS = seasonal_decompose(data_NS['Revenue_NS_normalized'], model='multiplicative', period=12)


        # Create a vector for monthly multipliers for FIN
        # Create a list of month names
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Calculate the monthly average for each year
        # Group by 'Year' and 'Month', and then calculate the mean for each month across all full years
        #monthly_avg_fin = full_years.groupby('Month')['Revenue FIN'].mean().reset_index()
        #monthly_avg_ind = full_years.groupby('Month')['Revenue IND'].mean().reset_index()
        #monthly_avg_ns = full_years.groupby('Month')['Revenue NS'].mean().reset_index()
    
        # Seasonal component is the same in the same calendar month across the years. The the first 12 months from the first January onwards.
        first_january_index = (self.train_data['Month'] == 1).idxmax()
        first_january_row = self.train_data.index.get_loc(first_january_index)
        self.monthly_multi_FIN = pd.DataFrame({'Month': months, 'Values': self.multiplicative_model_FIN.seasonal[first_january_row:first_january_row+12]})
        self.monthly_multi_IND = pd.DataFrame({'Month': months, 'Values': self.multiplicative_model_IND.seasonal[first_january_row:first_january_row+12]})
        self.monthly_multi_NS = pd.DataFrame({'Month': months, 'Values': self.multiplicative_model_NS.seasonal[first_january_row:first_january_row+12]})

        # Reset the index
        self.monthly_multi_FIN.reset_index(drop=True, inplace=True)
        self.monthly_multi_IND.reset_index(drop=True, inplace=True)
        self.monthly_multi_NS.reset_index(drop=True, inplace=True)

        # Drop nans from trend vectors
        self.cleaned_trend_FIN = self.multiplicative_model_FIN.trend.dropna()
        self.cleaned_trend_IND = self.multiplicative_model_IND.trend.dropna()
        self.cleaned_trend_NS = self.multiplicative_model_NS.trend.dropna()     

        # Drop nans from residual vectors
        self.cleaned_res_FIN = self.multiplicative_model_FIN.resid.dropna()
        self.cleaned_res_IND = self.multiplicative_model_IND.resid.dropna()
        self.cleaned_res_NS = self.multiplicative_model_NS.resid.dropna()   

        # Print residuals
        print(self.cleaned_res_FIN.describe())
        print(self.cleaned_res_IND.describe())
        print(self.cleaned_res_NS.describe())

    # Create forecasts for a given period. 
    # !!! The training (fuction train_model(...)) MUST HAVE DONE for the used forecast instance before forecasting.
    def forecast(self, forecast_start, forecast_end):    

        forecast_data = ForecastUtils.split_data(self.data, forecast_start, forecast_end)

        # Include working days in the forecast data
        forecast_data['Working_days_FIN'] = forecast_data['D/M FIN']
        forecast_data['Working_days_IND'] = forecast_data['D/M IND']
        forecast_data['Working_days_NS'] = forecast_data['D/M NS']

        #print(forecast_data)
        slope_fin, intercept_fin, r_value_fin, p_value_fin, std_err_fin = linregress(self.cleaned_trend_FIN.index, self.cleaned_trend_FIN.values)
        slope_ind, intercept_ind, r_value_ind, p_value_ind, std_err_ind = linregress(self.cleaned_trend_IND.index, self.cleaned_trend_IND.values)
        slope_ns, intercept_ns, r_value_ns, p_value_ns, std_err_ns = linregress(self.cleaned_trend_NS.index, self.cleaned_trend_NS.values)

        # Calcularte the range of calculation dynamically in the basis of the foreasting dates received as parameters
        forecast_start_month_index = int(forecast_data.iloc[0]['Month_index'])
        forecast_end_month_index = int(forecast_data.iloc[-1]['Month_index']) + 1
        months_to_predict = range(forecast_start_month_index, forecast_end_month_index)    
        #months_to_predict = range(33, 39)

        # Choose monthly multipliers dynamically in basis of the calendar month.
        # Running index in monthly_multi_FIN is used to match to the actual month in forecast_data.
        multis_per_predicted_month_fin = self.monthly_multi_FIN.iloc[
            [forecast_data.iloc[i]['Month'] - 1 for i in range(len(forecast_data))]
        ]
        multis_per_predicted_month_ind = self.monthly_multi_IND.iloc[
            [forecast_data.iloc[i]['Month'] - 1 for i in range(len(forecast_data))]
        ]
        multis_per_predicted_month_ns = self.monthly_multi_NS.iloc[
            [forecast_data.iloc[i]['Month'] - 1 for i in range(len(forecast_data))]
        ]

        # Seasonal components (we are predicting 4/24 - 9/24 so we take monthly multipliers indexed 3 to 8)
        # Multiplier can be found from vector monthly_multi_*
        #multis_per_predicted_month_fin = self.monthly_multi_FIN[3:9]        
        #multis_per_predicted_month_ind = self.monthly_multi_IND[3:9]    
        #multis_per_predicted_month_ns = self.monthly_multi_NS[3:9]    

        # Actual forecasting. To allow simple debugging and comparíng to the yupiter code, this has been left as-is.
        predictions_df_FIN = pd.DataFrame({
            'Month_index': months_to_predict,
            'Predictions': [(slope_fin * x + intercept_fin) * multis_per_predicted_month_fin.values[i] [1]
                        for i, x in enumerate(months_to_predict)]
        })
        predictions_df_IND = pd.DataFrame({
            'Month_index': months_to_predict,
            'Predictions': [(slope_ind * x + intercept_ind) * multis_per_predicted_month_ind.values[i] [1]
                        for i, x in enumerate(months_to_predict)]
        })
        predictions_df_NS = pd.DataFrame({
            'Month_index': months_to_predict,
            'Predictions': [(slope_ns * x + intercept_ns) * multis_per_predicted_month_ns.values[i] [1]
                        for i, x in enumerate(months_to_predict)]
        })

        # Scale predictions by working days
        predictions_df_FIN['Predictions'] *= forecast_data['Working_days_FIN'].values
        predictions_df_IND['Predictions'] *= forecast_data['Working_days_IND'].values
        predictions_df_NS['Predictions'] *= forecast_data['Working_days_NS'].values

        # Set index same for both so that we can sum these things
        #if 'Month_index' in predictions_df_FIN.columns:
        #    predictions_df_FIN = predictions_df_FIN.set_index('Month_index')
        #if 'Month_index' in predictions_df_IND.columns:
        #    predictions_df_IND = predictions_df_IND.set_index('Month_index')
        #if 'Month_index' in predictions_df_NS.columns:
        #    predictions_df_IND = predictions_df_NS.set_index('Month_index')
        
        # Create return vectors similar to other models to allow using data in common validation and evaluation functions.
        forecast_fin = predictions_df_FIN.set_index('Month_index')['Predictions']
        forecast_ind = predictions_df_IND.set_index('Month_index')['Predictions']
        forecast_ns = predictions_df_NS.set_index('Month_index')['Predictions']
        forecast_total = forecast_fin + forecast_ind + forecast_ns

        return forecast_fin, forecast_ind, forecast_ns, forecast_total

    # Utility function to validate the model. 
    # Just passing the data through to the common utility function in ForecastUtils.
    def validate(self, forecast_fin, forecast_ind, forecast_ns, validation_start, validation_end, save_errors=False, plot_errors=True):

        validation_data = ForecastUtils.split_data(self.data, validation_start, validation_end)
        
        # Change the index: Use validation_data's index to allow matching later in ForecastUtils.
        forecast_fin.index = validation_data.index
        forecast_ind.index = validation_data.index
        forecast_ns.index = validation_data.index

        result = ForecastUtils.validation_results(validation_data, forecast_fin, forecast_ind, forecast_ns, save_errors=save_errors, plot_errors=plot_errors, model_name='multicaptive_normalized', train_data=self.train_data)

        return result

# Usage example / dummy test
if __name__ == "__main__":

    # Instantiate the service
    service = RevenueForecastMulticaptive2() 

    # Train the model (it's internal submodels)
    service.train_model('2021-10-01', '2024-09-30') 

    # Forecasting
    forecast_fin, forecast_ind, forecast_ns, forecast_total = service.forecast('2024-11-01', '2025-11-30')
    # forecast_fin, forecast_ind, forecast_ns, forecast_total  = service.forecast('2024-08-01', '2024-08-30') 

    # service.validate(forecast_fin, forecast_ind, forecast_ns, '2024-08-01', '2024-08-30', save_errors=True)
    service.validate(forecast_fin, forecast_ind, forecast_ns, '2024-11-01', '2025-11-30', save_errors=True)
    