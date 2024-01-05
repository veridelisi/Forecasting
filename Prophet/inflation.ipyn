import requests
import pandas as pd
def get_fred_series_observations(series_id, api_key):
    # Endpoint for series observations
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json"
    }
    response = requests.get(base_url, params=params)
    return response.json()

api_key = 'ef8c3af7f7bebd62ffff5b460d66375a'
series_id = 'CPIAUCSL'

# Fetch the data points for the series
data = get_fred_series_observations(series_id, api_key)

# Check if observations are in the response and create a DataFrame
if 'observations' in data:
    df = pd.DataFrame(data['observations'])
    df = df[['date', 'value']]  # Select only the 'date' and 'value' columns
    df.rename(columns={'date': 'ds'}, inplace=True)


    df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert values to numeric, handling non-numeric entries
    df['ds'] = pd.to_datetime(df['ds'])  # Convert date strings to datetime objects

    # Calculate percent change from a year ago
    #df.set_index('ds', inplace=True)  # Set date column as index for easier manipulation
    df['y'] = (df['value'].pct_change(periods=12) * 100).round(2)  # Calculate percent change

else:
    print("Observations not found in the response.")
    df = pd.DataFrame()

pip install prophet
import pandas as pd
from prophet import Prophet
m = Prophet()
m.fit(df)
m = Prophet(seasonality_mode='multiplicative').fit(df)
future = m.make_future_dataframe(periods=30)
fcst = m.predict(future)
fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(fcst)
