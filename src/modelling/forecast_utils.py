import pandas as pd
import matplotlib.pyplot as plt

# Utility function to provide common functions. At this point, services include
# - load_data - load the datafile to dataframe
# - load_rate_normalized_data - load data and normalize revenues by rate raises
# - split_data_train_validation_forecast - split data to train, validation and test or forecast datasets in required intervals
# - split_data_train_validation - split data to train and validation sets in required date intervals
# - split_data - get a subset of data (between start and end dates)
# - validation_results - print errors to the screen + based on parameters save to a csv file and/or plot
# - plot
class ForecastUtils:
    @staticmethod
    def load_data():
        df = pd.read_csv('data.csv')

        # Create DateTime indices
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
        df.set_index('Date', inplace=True)
        df.index.freq = 'MS' 
    
        # Ensure that decimal fields are correct and numeric
        df['Revenue FIN'] = pd.to_numeric(df['Revenue FIN'], errors='raise')
        df['Revenue IND'] = pd.to_numeric(df['Revenue IND'], errors='raise')
        df['Revenue NS'] = pd.to_numeric(df['Revenue NS'], errors='raise')
        #df['Revenue IND'] = df['Revenue IND'].astype(float)
        df['D/M FIN'] = pd.to_numeric(df['D/M FIN'], errors='raise')
        df['D/M IND'] = pd.to_numeric(df['D/M IND'], errors='raise')
        df['D/M NS'] = pd.to_numeric(df['D/M NS'], errors='raise')
           
        return df
    
    @staticmethod
    def load_rate_normalized_data():
        df = ForecastUtils.load_data()

        # Create a raise column from rate raise information in individual month information. 
        # This is a multiplier that tells how much data rates have changed from the start
        # of the file (multiplier 1.1 = rates have raised by 10%).
        multiplier_fin = 1.0
        multiplier_ind = 1.0
        multiplier_ns = 1.0
        for i, row in df.iterrows():
            raise_fin = float(row['Raise FIN'].rstrip('%'))
            raise_ind = float(row['Raise IND'].rstrip('%'))
            raise_ns = float(row['Raise NS'].rstrip('%'))

            if(raise_fin != 0):
                multiplier_fin = multiplier_fin * (1 + raise_fin/100)
            if(raise_ind != 0):
                multiplier_ind = multiplier_ind * (1 + raise_ind/100)
            if(raise_ns != 0):
                multiplier_ns = multiplier_ns * (1 + raise_ns/100)            

            df.at[i, 'raise_multiplier_fin'] = multiplier_fin
            df.at[i, 'raise_multiplier_ind'] = multiplier_ind
            df.at[i, 'raise_multiplier_ns'] = multiplier_ns

        # Normalize historical revenues by raise multipliers.
        # This removes effect of rate raises when calculating cyclicity and 
        # leaves only true change in trend (actual volume change) to numbers.
        for i, row in df.iterrows():
            revenue_fin = multiplier_fin/row['raise_multiplier_fin'] * row['Revenue FIN']
            #print(f"original revenue ' {row['Year']} {row['Month']} {row['Revenue FIN']}, {revenue_fin}")
                
            revenue_ind = multiplier_fin / row['raise_multiplier_ind'] * row['Revenue IND']
            revenue_ns = multiplier_ns / row['raise_multiplier_ns'] * row['Revenue NS']
            df.at[i, 'Revenue FIN'] = revenue_fin
            df.at[i, 'Revenue IND'] = revenue_ind
            df.at[i, 'Revenue NS'] = revenue_ns    

        return df
    
    # Split the data to three sets: training, validation and test or true forecast.
    # This method expects that there is a date index created in the dataset. If using data load methods of this utility class, it's already done there.
    @staticmethod
    def split_data_train_validation_forecast(df, train_start, train_end, validatation_start, validatation_end, forecast_start, forecast_end):

        train_data = df[(df.index >= train_start) & (df.index <= train_end)]
        validation_data = df[(df.index >= validatation_start) & (df.index <= validatation_end)]
        forecast_data = df[(df.index >= forecast_start) & (df.index <= forecast_end)]

        return train_data, validation_data, forecast_data
    
    # Split the data to training and validation sets
    # This method expects that there is a date index created in the dataset. If using data load methods of this utility class, it's already done there.
    @staticmethod    
    def split_data_train_validation(df, train_start, train_end, validatation_start, validatation_end):

        train_data = df[(df.index >= train_start) & (df.index <= train_end)]
        validation_data = df[(df.index >= validatation_start) & (df.index <= validatation_end)]

        return train_data, validation_data
    
    # Return a subset of data between start and end dates.
    # This method expects that there is a date index created in the dataset. If using data load methods of this utility class, it's already done there.
    @staticmethod    
    def split_data(df, start_date, end_date):
        data = df[(df.index >= start_date) & (df.index <= end_date)]
        return data
    
    # Return a subset of data between start and end dates.
    # This method expects that there is a date index created in the dataset. If using data load methods of this utility class, it's already done there.
    @staticmethod    
    def validation_results(validation_data, forecast_FIN, forecast_IND, forecast_NS, model_name='model', save_errors=False, plot_errors=False, train_data=None):
      
        predict_total = forecast_FIN + forecast_IND + forecast_NS
        actual_total = validation_data['Revenue FIN'] + validation_data['Revenue IND'] + validation_data['Revenue NS']

        # Create the DataFrame to compare actual and forecasted revenues
        monthly_errors = pd.DataFrame({
            'Actual Revenue FIN': validation_data['Revenue FIN'].apply(lambda x: f'{int(x):,d}'),
            'Forecasted Revenue FIN': pd.Series(forecast_FIN, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
            'Actual Revenue IND': validation_data['Revenue IND'].apply(lambda x: f'{int(x):,d}'),
            'Forecasted Revenue IND': pd.Series(forecast_IND, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
            'Actual Revenue NS': validation_data['Revenue NS'].apply(lambda x: f'{int(x):,d}'),
            'Forecasted Revenue NS': pd.Series(forecast_NS, index=validation_data.index).apply(lambda x: f'{int(x):,d}'),
            'Actual Total Revenue': actual_total.apply(lambda x: f'{int(x):,d}'),
            'Forecasted Total Revenue': pd.Series(predict_total, index=validation_data.index).apply(lambda x: f'{int(x):,d}')
        })

        # Add Error% FIN, Error% IND, Error% NS and Total Revenue Error% columns
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
        monthly_errors['Error% Total'] = monthly_errors.apply(
            lambda row: None if row['Actual Total Revenue'] is None else 
            f"{((float(row['Actual Total Revenue'].replace(',', '')) - float(row['Forecasted Total Revenue'].replace(',', ''))) / float(row['Actual Total Revenue'].replace(',', '')) * 100):.2f}%", axis=1
        )    

        print(monthly_errors)

        # Print errors to csv file for further use if requested
        if(save_errors):
            filename = model_name + '_errors.csv'
            monthly_errors.to_csv(filename)

        if(plot_errors):
            ForecastUtils.plot_results(train_data, validation_data, forecast_FIN, forecast_IND, forecast_NS)

    @staticmethod  
    def plot_results(train_data, validation_data, forecast_FIN, forecast_IND, forecast_NS):  
                                                                                                        
        # Plot forecasted vs actual Revenue IND and Revenue FIN, including the extended forecast period
        plt.figure(figsize=(10, 6))

        # Plot training data
        if train_data is not None and not train_data.empty:
            plt.plot(train_data.index, train_data['Revenue FIN'], label='Actual Revenue FIN (Training)', color='blue', linestyle='--')
            plt.plot(train_data.index, train_data['Revenue IND'], label='Actual Revenue IND (Training)', color='green', linestyle='--')

        # Plot validation data
        plt.plot(validation_data.index, validation_data['Revenue FIN'], label='Actual Revenue FIN', color='blue')
        plt.plot(validation_data.index, forecast_FIN, label='Forecasted Revenue FIN', linestyle='--', color='red')
        plt.plot(validation_data.index, validation_data['Revenue IND'], label='Actual Revenue IND', color='green')        
        plt.plot(validation_data.index, forecast_IND, label='Forecasted Revenue IND', linestyle='--', color='orange')

        plt.title('Forecasted vs Actual Revenue for IND and FIN (Including Extended Forecast Period)')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.legend()
        plt.tight_layout()
        plt.show()