import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
prices = pd.read_csv(
    "data/raw_prices.csv",
    index_col=0,        # first column is Date
    parse_dates=True   # convert Date to datetime
)

prices.plot(figsize=(10, 5), title="Adjusted Stock Prices")
plt.show()
returns  = np.log(prices/prices.shift(1)) # This shifts everything down by 1 row:
returns = returns.dropna()

returns["AAPL"].plot(title="AAPL Daily Log Returns")
plt.show()