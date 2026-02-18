
import pandas as pd
import numpy as np
import pandas_datareader.data as web

class DataHandler:
    def __init__(self, price_path):
        self.prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        self.returns = None
        self.train = None
        self.test = None

    def compute_returns(self,method="simple"):
        if method == "simple":
            self.returns = self.prices.pct_change().dropna()
        elif method == "log":
            self.returns  = np.log(self.prices/self.prices.shift(1))
        self.returns = self.returns.dropna()
        return self.returns
    
    def test_train_split(self,split_date):
        if self.returns is None:
            raise RuntimeError("Call compute_returns() first")
        self.train = self.returns.loc[:split_date]
        self.test = self.returns.loc[split_date:]
        return self.train, self.test

    def get_risk_free_rate(self, start, end):
        rf = web.DataReader("DTB3", "fred", start, end)
        rf = rf.dropna()
        return (rf["DTB3"] / 100).mean()
