import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "JNJ", "PG", "XOM", "CAT", "VZ"]
start_date = "2010-01-01"
end_date = "2024-12-31"

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

close  = data["Close"]
high   = data["High"]
low    = data["Low"]
volume = data["Volume"]   

common_index = close.dropna().index
common_index = common_index.intersection(high.dropna().index)
common_index = common_index.intersection(low.dropna().index)
common_index = common_index.intersection(volume.dropna().index)

close  = close.loc[common_index]
high   = high.loc[common_index]
low    = low.loc[common_index]
volume = volume.loc[common_index]


spread_proxy = (high - low) / close


close.to_csv("data/close_prices.csv")
volume.to_csv("data/volume.csv")         
spread_proxy.to_csv("data/spread_proxy.csv")
