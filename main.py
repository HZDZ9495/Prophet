import pandas as pd
import prophet as Prophet
from prophet.plot import plot_plotly
import plotly.io as pio

# data = pd.read_csv('sunspots.csv')
data = pd.read_csv('sunspots-after1900.csv')
df = pd.DataFrame()
df['ds'] = pd.to_datetime(data['Month'])
df['y'] = data['Sunspots']

model = Prophet(yearly_seasonality=0, weekly_seasonality=0, daily_seasonality=0,growth='logistic')
model.add_seasonality(name='11years', period=365.25*11.2, fourier_order=15,mode='multiplicative')
model.add_seasonality(name='monthly', period=30.5, fourier_order=5,mode='multiplicative')

df['cap'] = 300 
df['floor'] = 0
model.fit(df)

future_dates = model.make_future_dataframe(periods=500, freq='M') 
future_dates['cap'] = 300
future_dates['floor'] = 0

forecast = model.predict(future_dates)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig = plot_plotly(model, forecast)
pio.show(fig)