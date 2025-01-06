import pandas as pd
from revenue_forecast_runrate import RevenueForecastRunrate
from revenue_forecast_sarimax import RevenueForecastSarimax
from revenue_forecast_multicaptive import RevenueForecastMulticaptive
from revenue_forecast_multicaptive_normalized import (
    RevenueForecastMulticaptiveNormalized,
)


# Compare 3 models in a 6 month validation period (Apr-Sep 2024)
# Two comparisons are included:
# - full period with one training
# - rolling validation for each month in same period, re-training with all of the preceding data for each iteration
def model_comparison():

    # Normal validation
    sarimax_model = RevenueForecastSarimax()
    runrate_model_without_trend = RevenueForecastRunrate()
    runrate_model_with_trend = RevenueForecastRunrate(use_trend=True)
    multicaptive_model = RevenueForecastMulticaptive()
    multicaptive_model_normalized = RevenueForecastMulticaptiveNormalized()

    training_start = "2021-10-01"
    training_end = "2024-10-31"
    sarimax_model.train_model(training_start, training_end)
    runrate_model_without_trend.train_model(training_start, training_end)
    runrate_model_with_trend.train_model(training_start, training_end)
    multicaptive_model.train_model(training_start, training_end)
    multicaptive_model_normalized.train_model(training_start, training_end)

    forecast_start = "2024-11-01"
    forecast_end = "2024-12-31"
    forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_total_sm = (
        sarimax_model.forecast(forecast_start, forecast_end)
    )
    forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_total_rn = (
        runrate_model_without_trend.forecast(forecast_start, forecast_end)
    )
    forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_total_sm = (
        runrate_model_with_trend.forecast(forecast_start, forecast_end)
    )
    forecast_fin_mc, forecast_ind_mc, forecast_ns_mc, forecast_total_mc = (
        multicaptive_model.forecast(forecast_start, forecast_end)
    )
    forecast_fin_mn, forecast_ind_mn, forecast_ns_mn, forecast_total_mn = (
        multicaptive_model_normalized.forecast(forecast_start, forecast_end)
    )

    sarimax_model.validate(
        forecast_fin_sm,
        forecast_ind_sm,
        forecast_ns_sm,
        forecast_start,
        forecast_end,
        save_errors=True,
    )
    runrate_model_without_trend.validate(
        forecast_fin_rn,
        forecast_ind_rn,
        forecast_ns_rn,
        forecast_start,
        forecast_end,
        save_errors=True,
    )
    runrate_model_with_trend.validate(
        forecast_fin_rt,
        forecast_ind_rt,
        forecast_ns_rt,
        forecast_start,
        forecast_end,
        save_errors=True,
    )
    multicaptive_model.validate(
        forecast_fin_mc,
        forecast_ind_mc,
        forecast_ns_mc,
        forecast_start,
        forecast_end,
        save_errors=True,
    )
    multicaptive_model_normalized.validate(
        forecast_fin_mn,
        forecast_ind_mn,
        forecast_ns_mn,
        forecast_start,
        forecast_end,
        save_errors=True,
    )


# Rolling validation
# To remove risk of any interference, always create a new model instance for each round even if the same instance with re-training should work
def model_comparison_rolling_validation():

    rolling_validation_periods = [
        ["2024-10-31", "2024-11-01", "2024-11-30"],
        ["2024-11-30", "2024-12-01", "2024-12-31"],
    ]

    # Loop through the rolling validation periods and collect results
    results = []
    csv_data = []
    for period in rolling_validation_periods:

        training_start = "2021-10-01"
        training_end = period[0]
        forecast_start = period[1]
        forecast_end = period[2]

        # Initialize models
        sarimax_model = RevenueForecastSarimax()
        runrate_model_without_trend = RevenueForecastRunrate()
        runrate_model_with_trend = RevenueForecastRunrate(use_trend=True)
        multicaptive_model = RevenueForecastMulticaptive()
        multicaptive_model_normalized = RevenueForecastMulticaptiveNormalized()

        # Train models
        sarimax_model.train_model(training_start, training_end)
        runrate_model_without_trend.train_model(training_start, training_end)
        runrate_model_with_trend.train_model(training_start, training_end)
        multicaptive_model.train_model(training_start, training_end)
        multicaptive_model_normalized.train_model(training_start, training_end)

        # Create validation forecasts
        forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_total_sm = (
            sarimax_model.forecast(forecast_start, forecast_end)
        )
        forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_total_rn = (
            runrate_model_without_trend.forecast(forecast_start, forecast_end)
        )
        forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_total_sm = (
            runrate_model_with_trend.forecast(forecast_start, forecast_end)
        )
        forecast_fin_mc, forecast_ind_mc, forecast_ns_mc, forecast_total_mc = (
            multicaptive_model.forecast(forecast_start, forecast_end)
        )
        forecast_fin_mn, forecast_ind_mn, forecast_ns_mn, forecast_total_mn = (
            multicaptive_model_normalized.forecast(forecast_start, forecast_end)
        )

        results_sm = sarimax_model.validate(
            forecast_fin_sm,
            forecast_ind_sm,
            forecast_ns_sm,
            forecast_start,
            forecast_end,
            save_errors=False,
        )
        results_rn = runrate_model_without_trend.validate(
            forecast_fin_rn,
            forecast_ind_rn,
            forecast_ns_rn,
            forecast_start,
            forecast_end,
            save_errors=False,
        )
        results_rt = runrate_model_with_trend.validate(
            forecast_fin_rt,
            forecast_ind_rt,
            forecast_ns_rt,
            forecast_start,
            forecast_end,
            save_errors=False,
        )
        results_mc = multicaptive_model.validate(
            forecast_fin_mc,
            forecast_ind_mc,
            forecast_ns_mc,
            forecast_start,
            forecast_end,
            save_errors=False,
        )
        results_mn = multicaptive_model_normalized.validate(
            forecast_fin_mn,
            forecast_ind_mn,
            forecast_ns_mn,
            forecast_start,
            forecast_end,
            save_errors=False,
        )

        results.append([results_sm, results_rn, results_rt, results_mc, results_mn])

        # Add the monthly results to csv_data
        for model_name, model_results in zip(
            [
                "SARIMAX",
                "Runrate Notrend",
                "Runrate Trend",
                "Multicaptive",
                "Multicaptive Normalized",
            ],
            [results_sm, results_rn, results_rt, results_mc, results_mn],
        ):
            csv_data.append(
                {
                    "Period": period[1],
                    "Model": model_name,
                    "Total Error (%)": model_results[0],
                    "FIN Error (%)": model_results[1],
                    "IND Error (%)": model_results[2],
                    "NS Error (%)": model_results[3],
                }
            )

    # Print the results for each month
    print("Rolling validation monthly results")
    for i, period in enumerate(rolling_validation_periods):
        date = period[1]
        print(f"Month {date}")
        for model_name, model_results in zip(
            [
                "SARIMAX",
                "Runrate Notrend",
                "Runrate Trend",
                "Multicaptive",
                "Multicaptive Normalized",
            ],
            results[i],
        ):
            print(f"  {model_name}:")
            print(f"    Total error: {model_results[0]:,.1f}%")
            print(f"    FIN error: {model_results[1]:,.1f}%")
            print(f"    IND error: {model_results[2]:,.1f}%")
            print(f"    NS error: {model_results[3]:,.1f}%")

    # Calculate averages over validation period
    total_errors_sm = [0, 0, 0, 0]  # For SARIMAX model
    total_errors_rn = [0, 0, 0, 0]  # For Runrate without trend
    total_errors_rt = [0, 0, 0, 0]  # For Runrate with trend
    total_errors_mc = [0, 0, 0, 0]  # For Multicaptive
    total_errors_mn = [0, 0, 0, 0]  # For Multicaptive normalized
    num_periods = len(rolling_validation_periods)
    for i in range(num_periods):
        total_errors_sm = [total_errors_sm[j] + abs(results[i][0][j]) for j in range(4)]
        total_errors_rn = [total_errors_rn[j] + abs(results[i][1][j]) for j in range(4)]
        total_errors_rt = [total_errors_rt[j] + abs(results[i][2][j]) for j in range(4)]
        total_errors_mc = [total_errors_mc[j] + abs(results[i][3][j]) for j in range(4)]
        total_errors_mn = [total_errors_mn[j] + abs(results[i][4][j]) for j in range(4)]
    avg_errors_sm = [error / num_periods for error in total_errors_sm]
    avg_errors_rn = [error / num_periods for error in total_errors_rn]
    avg_errors_rt = [error / num_periods for error in total_errors_rt]
    avg_errors_mc = [error / num_periods for error in total_errors_mc]
    avg_errors_mn = [error / num_periods for error in total_errors_mn]

    print(
        "\nRolling validation results for the whole validation period (Average absolute monthly errors over the whole period):"
    )
    for model_name, avg_errors in zip(
        [
            "SARIMAX",
            "Runrate Notrend",
            "Runrate Trend",
            "Multicaptive",
            "Multicaptive Normalized",
        ],
        [avg_errors_sm, avg_errors_rn, avg_errors_rt, avg_errors_mc, avg_errors_mn],
    ):
        print(f"\n{model_name}:")
        print(
            f"Mean of total error absolute values in the validation period: {avg_errors[0]:,.1f}%"
        )
        print(
            f"Mean of FIN error absolute values in the validation period: {avg_errors[1]:,.1f}%"
        )
        print(
            f"Mean of IND error absolute values in the validation period: {avg_errors[2]:,.1f}%"
        )
        print(
            f"Mean of NS error absolute values in the validation period: {avg_errors[3]:,.1f}%"
        )

    # Save results to CSV
    csv_file_path = "rolling_validation_results.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file_path, index=False)
    print(f"\nRolling validation results saved to {csv_file_path}")


# Create a 12 month FC from all models
def forecast_12_months():

    # Normal validation
    sarimax_model = RevenueForecastSarimax()
    runrate_model_without_trend = RevenueForecastRunrate()
    runrate_model_with_trend = RevenueForecastRunrate(use_trend=True)
    multicaptive_model = RevenueForecastMulticaptive()
    multicaptive_model_normalized = RevenueForecastMulticaptiveNormalized()

    training_start = "2021-10-01"
    training_end = "2024-09-30"

    sarimax_model.train_model(training_start, training_end)
    runrate_model_without_trend.train_model(training_start, training_end)
    runrate_model_with_trend.train_model(training_start, training_end)
    multicaptive_model.train_model(training_start, training_end)
    multicaptive_model_normalized.train_model(training_start, training_end)

    forecast_start = "2024-10-01"
    forecast_end = "2025-09-30"
    forecast_fin_sm, forecast_ind_sm, forecast_ns_sm, forecast_total_sm = (
        sarimax_model.forecast(forecast_start, forecast_end)
    )
    forecast_fin_rn, forecast_ind_rn, forecast_ns_rn, forecast_total_rn = (
        runrate_model_without_trend.forecast(forecast_start, forecast_end)
    )
    forecast_fin_rt, forecast_ind_rt, forecast_ns_rt, forecast_total_sm = (
        runrate_model_with_trend.forecast(forecast_start, forecast_end)
    )
    forecast_fin_mc, forecast_ind_mc, forecast_ns_mc, forecast_total_mc = (
        multicaptive_model.forecast(forecast_start, forecast_end)
    )
    forecast_fin_mn, forecast_ind_mn, forecast_ns_mn, forecast_total_mn = (
        multicaptive_model_normalized.forecast(forecast_start, forecast_end)
    )

    # Ensure all forecasts are lists
    forecast_fin_sm = (
        forecast_fin_sm.tolist()
        if isinstance(forecast_fin_sm, pd.Series)
        else forecast_fin_sm
    )
    forecast_ind_sm = (
        forecast_ind_sm.tolist()
        if isinstance(forecast_ind_sm, pd.Series)
        else forecast_ind_sm
    )
    forecast_ns_sm = (
        forecast_ns_sm.tolist()
        if isinstance(forecast_ns_sm, pd.Series)
        else forecast_ns_sm
    )

    forecast_fin_rn = (
        forecast_fin_rn.tolist()
        if isinstance(forecast_fin_rn, pd.Series)
        else forecast_fin_rn
    )
    forecast_ind_rn = (
        forecast_ind_rn.tolist()
        if isinstance(forecast_ind_rn, pd.Series)
        else forecast_ind_rn
    )
    forecast_ns_rn = (
        forecast_ns_rn.tolist()
        if isinstance(forecast_ns_rn, pd.Series)
        else forecast_ns_rn
    )

    forecast_fin_rt = (
        forecast_fin_rt.tolist()
        if isinstance(forecast_fin_rt, pd.Series)
        else forecast_fin_rt
    )
    forecast_ind_rt = (
        forecast_ind_rt.tolist()
        if isinstance(forecast_ind_rt, pd.Series)
        else forecast_ind_rt
    )
    forecast_ns_rt = (
        forecast_ns_rt.tolist()
        if isinstance(forecast_ns_rt, pd.Series)
        else forecast_ns_rt
    )

    forecast_fin_mc = (
        forecast_fin_mc.tolist()
        if isinstance(forecast_fin_mc, pd.Series)
        else forecast_fin_mc
    )
    forecast_ind_mc = (
        forecast_ind_mc.tolist()
        if isinstance(forecast_ind_mc, pd.Series)
        else forecast_ind_mc
    )
    forecast_ns_mc = (
        forecast_ns_mc.tolist()
        if isinstance(forecast_ns_mc, pd.Series)
        else forecast_ns_mc
    )

    forecast_fin_mn = (
        forecast_fin_mn.tolist()
        if isinstance(forecast_fin_mn, pd.Series)
        else forecast_fin_mn
    )
    forecast_ind_mn = (
        forecast_ind_mn.tolist()
        if isinstance(forecast_ind_mn, pd.Series)
        else forecast_ind_mn
    )
    forecast_ns_mn = (
        forecast_ns_mn.tolist()
        if isinstance(forecast_ns_mn, pd.Series)
        else forecast_ns_mn
    )

    # Create a list of months for columns
    months = (
        pd.date_range(start=forecast_start, end=forecast_end, freq="M")
        .strftime("%Y-%m")
        .tolist()
    )

    # Prepare data for each model and location
    forecast_data = {
        "Model": ["SARIMAX"] * 3
        + ["Runrate without Trend"] * 3
        + ["Runrate with Trend"] * 3
        + ["Multicaptive"] * 3
        + ["Multicaptive Normalized"] * 3,
        "Location": ["FIN", "IND", "NS"] * 5,
    }

    # Add month forecasts to forecast_data dictionary
    for i, month in enumerate(months):
        forecast_data[month] = [
            forecast_fin_sm[i],
            forecast_ind_sm[i],
            forecast_ns_sm[i],
            forecast_fin_rn[i],
            forecast_ind_rn[i],
            forecast_ns_rn[i],
            forecast_fin_rt[i],
            forecast_ind_rt[i],
            forecast_ns_rt[i],
            forecast_fin_mc[i],
            forecast_ind_mc[i],
            forecast_ns_mc[i],
            forecast_fin_mn[i],
            forecast_ind_mn[i],
            forecast_ns_mn[i],
        ]

    # Create a DataFrame from the forecast data
    df = pd.DataFrame(forecast_data)

    # Save to CSV file
    csv_file_path = "12_month_forecast.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"12-month forecast saved to {csv_file_path}")


# model_comparison_rolling_validation()
model_comparison()
# forecast_12_months()
