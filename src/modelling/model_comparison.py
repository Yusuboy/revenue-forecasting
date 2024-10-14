from revenue_forecast_runrate import RevenueForecastRunrate 
from revenue_forecast_sarimax import RevenueForecastSarimax

# Compare 3 models in a 6 month validation period (Apr-Sep 2024)
# Two comparisons are included: 
# - full period with one training
# - rolling validation for each month in same period, re-training with all of the preceding data for each iteration
def model_comparison():

    # Normal validation
    sarimax_model = RevenueForecastSarimax()
    runrate_model_without_trend = RevenueForecastRunrate()
    runrate_model_with_trend = RevenueForecastRunrate(use_trend=True)

    training_start = '2021-10-01'
    training_end = '2024-03-31'
    sarimax_model.train_model(training_start, training_end) 
    runrate_model_without_trend.train_model(training_start, training_end) 
    runrate_model_with_trend.train_model(training_start, training_end) 

    forecast_start = '2024-04-01'
    forecast_end = '2024-09-30'
    forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_total_sm = sarimax_model.forecast(forecast_start, forecast_end) 
    forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_total_rn = runrate_model_without_trend.forecast(forecast_start, forecast_end) 
    forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_total_sm = runrate_model_with_trend.forecast(forecast_start, forecast_end) 

    sarimax_model.validate(forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_start, forecast_end)
    runrate_model_without_trend.validate(forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_start, forecast_end)
    runrate_model_with_trend.validate(forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_start, forecast_end)

# Rolling validation
# To remove risk of any interference, always create a new model instance for each round even if the same instance with re-training should work
def model_comparison_rolling_validatio():

    rolling_validation_periods = [
        ['2024-03-31', '2024-04-01', '2024-04-30'],
        ['2024-04-30', '2024-05-01', '2024-05-31'],
        ['2024-05-31', '2024-06-01', '2024-06-30'],
        ['2024-06-30', '2024-07-01', '2024-07-31'],
        ['2024-07-31', '2024-08-01', '2024-08-31'],
        ['2024-08-31', '2024-09-01', '2024-09-30']        
    ]

    # Loop through the rolling validation periods and collect results
    results = []
    for period in rolling_validation_periods:

        training_start = '2021-10-01'
        training_end = period[0]
        forecast_start = period[1]
        forecast_end = period[2]
        sarimax_model = RevenueForecastSarimax()
        runrate_model_without_trend = RevenueForecastRunrate()
        runrate_model_with_trend = RevenueForecastRunrate(use_trend=True)

        sarimax_model.train_model(training_start, training_end) 
        runrate_model_without_trend.train_model(training_start, training_end) 
        runrate_model_with_trend.train_model(training_start, training_end) 

        forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_total_sm = sarimax_model.forecast(forecast_start, forecast_end) 
        forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_total_rn = runrate_model_without_trend.forecast(forecast_start, forecast_end) 
        forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_total_sm = runrate_model_with_trend.forecast(forecast_start, forecast_end)     

        results_sm = sarimax_model.validate(forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_start, forecast_end)
        results_rn = runrate_model_without_trend.validate(forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_start, forecast_end)
        results_rt = runrate_model_with_trend.validate(forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_start, forecast_end)   

        results.append([results_sm, results_rn, results_rt])

    # Print the results for each month
    print('Rolling validation monthly results')
    for i, period in enumerate(rolling_validation_periods):
        date = period[1]
        print(f'Month {date}')
        for model_name, model_results in zip(['SARIMAX', 'Runrate Without Trend', 'Runrate With Trend'], results[i]):
            print(f"  {model_name}:")
            print(f"    Total error: {model_results[0]:,.1f}%")
            print(f"    FIN error: {model_results[1]:,.1f}%")
            print(f"    IND error: {model_results[2]:,.1f}%")
            print(f"    NS error: {model_results[3]:,.1f}%")

    # Calculate averages over validation period
    total_errors_sm = [0, 0, 0, 0]  # For SARIMAX model
    total_errors_rn = [0, 0, 0, 0]  # For Runrate without trend
    total_errors_rt = [0, 0, 0, 0]  # For Runrate with trend
    num_periods = len(rolling_validation_periods)
    for i in range(num_periods):
        total_errors_sm = [total_errors_sm[j] + abs(results[i][0][j]) for j in range(4)]
        total_errors_rn = [total_errors_rn[j] + abs(results[i][1][j]) for j in range(4)]
        total_errors_rt = [total_errors_rt[j] + abs(results[i][2][j]) for j in range(4)]
    avg_errors_sm = [error / num_periods for error in total_errors_sm]
    avg_errors_rn = [error / num_periods for error in total_errors_rn]
    avg_errors_rt = [error / num_periods for error in total_errors_rt]

    print("\nRolling validation results for the whole validation period (Average absolute monthly errors over the whole period):")
    for model_name, avg_errors in zip(['SARIMAX', 'Runrate Without Trend', 'Runrate With Trend'], [avg_errors_sm, avg_errors_rn, avg_errors_rt]):
        print(f"\n{model_name}:")
        print(f'Mean of total error absolute values in the validation period: {avg_errors[0]:,.1f}%')
        print(f'Mean of FIN error absolute values in the validation period: {avg_errors[1]:,.1f}%')
        print(f'Mean of IND error absolute values in the validation period: {avg_errors[2]:,.1f}%')
        print(f'Mean of NS error absolute values in the validation period: {avg_errors[3]:,.1f}%')

model_comparison_rolling_validatio()