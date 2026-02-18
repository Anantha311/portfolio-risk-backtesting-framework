import pandas as pd
from data.data_handler import DataHandler
from optimization.portfolio_optimizer import PortfolioOptimizer

class RollingBacktester:
    def __init__(self, price_path,train_period=6,rollback_period=3): # Give the entire returns, and roll back period is in months
        self.prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        self.data = DataHandler(price_path)
        self.returns = self.data.compute_returns()
        new_returns = self.returns.sort_index()
        self.first_date = pd.to_datetime(new_returns.index[0].strftime('%Y-%m-%d'))
        self.last_date  = pd.to_datetime(new_returns.index[-1].strftime('%Y-%m-%d'))
        self.roll_period = pd.DateOffset(months=rollback_period)
        self.train_period = pd.DateOffset(months=train_period)
        self.test_period = self.roll_period 
        self.strategy_returns = {}

    def compute(self):
        current_date = self.first_date
        strategy_returns = {}
        while True:
            if current_date + self.train_period + self.test_period > self.last_date:  
                break
            else :
                train_start = current_date
                train_end   = current_date + self.train_period
                test_start  = train_end + pd.DateOffset(days=1)
                test_end    = test_start + self.test_period
                train_data = self.returns.loc[train_start:train_end] # .loc is for rows and coloums and .iloc is for specific values
                test_data = self.returns.loc[test_start:test_end] # What pandas does (important), .loc[start:end] on a DatetimeIndex: does NOT require start to exist it finds the first index ≥ start and the last index ≤ end
                if train_data.empty or test_data.empty:
                    current_date += self.roll_period
                    continue
                weights = self.get_weights(train_data,start_date=train_start,end_date=train_end)
                temp_portfolio_returns = self.compute_portfolio_returns(test_data,weights)
                for name, returns in temp_portfolio_returns.items():
                    if name not in strategy_returns:
                        strategy_returns[name] = []
                    strategy_returns[name].append(returns)         
                current_date = current_date + self.roll_period 
        for name, returns in strategy_returns.items():
            self.strategy_returns[name] = (pd.concat(strategy_returns[name]).sort_index().loc[~pd.concat(strategy_returns[name]).index.duplicated(keep="first")])
        return self.strategy_returns       

    def get_weights(self,train_data,start_date,end_date):
        optimizer = PortfolioOptimizer(train_data)
        risk_free_rate = self.data.get_risk_free_rate(start_date,end_date )
        return {
        "Tangency": optimizer.tangency(risk_free_rate),
        "Min Variance": optimizer.min_variance(),
        "Equal Weight": optimizer.equal_weight(),
        "Random": optimizer.random()[0]
        }

    def compute_portfolio_returns(self, test_returns,weights):
        portfolio_returns = {}
        for name, weight in weights.items():
            if not isinstance(weight, pd.Series): # So as to check if weight is numpy array or pandas array, if it is numpy array then numpy arrays are not named by colomuns whereas Pandas aligns by labels (index / column names), not by position. so AAPL will have certain weight and AAPL test_returns will have returns say they are in pos =3 in weights and pos=9 in test_returns still Pandas will multiply them coorectly as they are namned so this is extremely helpfull if later the order gets shuffled, numpy will always follow the matrix multiplication method as there column name is not present 
                weight = pd.Series(weight, index=test_returns.columns)
            portfolio_returns[name] = test_returns @ weight

        return portfolio_returns
