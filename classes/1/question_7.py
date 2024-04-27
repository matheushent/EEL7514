import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('classes/1/coin_Bitcoin.csv')

df['Date'] = pd.to_datetime(df['Date'])

df_2020 = df[df['Date'].dt.year == 2020]

dates = df_2020['Date'].values
close_prices = df_2020['Close'].values

plt.figure(figsize=(10, 5))
plt.plot(dates, close_prices, label='Closing Price')
plt.title('Closing Price do Bitcoin em 2020')
plt.xlabel('Data')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
