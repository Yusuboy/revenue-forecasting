from revenue_forecast_runrate import RevenueForecastRunrate 
from revenue_forecast_sarimax import RevenueForecastSarimax

# Compare 3 models in a 6 month validation period (Apr-Sep 2024)
# Two comparisons are included: 
# - full period with one training
# - rolling validation for each month in same period, re-training with all of the preceding data for each iteration
def model_comparison():

    # Normali validation
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

model_comparison()