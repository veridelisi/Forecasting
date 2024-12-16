from datetime import datetime

# Assuming the last date in your dataset as an example
last_date_in_dataset = df.index[-1]  # The last date in your dataset
target_date = datetime(2025, 3, 31)  # March 2025

# Calculate the number of months to forecast
months_to_forecast = (target_date.year - last_date_in_dataset.year) * 12 + target_date.month - last_date_in_dataset.month

# Adjust `nobs` if your last dataset date is before the target date
if months_to_forecast > 0:
    # You may need to retrain your model if the dataset has been updated
    # Assuming `model_fitted` is your last trained model and is still valid

    # Forecast the additional months needed
    fc_additional = model_fitted.forecast(y=forecast_input, steps=months_to_forecast)
    df_forecast_additional = pd.DataFrame(fc_additional, index=pd.date_range(start=last_date_in_dataset + pd.offsets.MonthBegin(1), periods=months_to_forecast, freq='M'), columns=df.columns + '_d')

    # Append the new forecasts to your existing forecast dataframe if needed
    df_forecast_combined = pd.concat([df_forecast, df_forecast_additional])

    # Now invert transformation on the combined forecast dataframe
    df_results_combined = invert_transformation(df_train, df_forecast_combined, second_diff=False)



# Filter the DataFrame to include rows with index greater than '2024-11-01'
filtered_forecast = df_results_combined[df_results_combined.index > '2024-11-01']

# Extract the CPI forecast column
cpi_forecast_after_nov2024 = filtered_forecast['CPI_forecast']

# Print the filtered forecast
print(cpi_forecast_after_nov2024)
