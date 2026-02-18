import pandas as pd
from data.data_handler import DataHandler
from optimization.portfolio_optimizer import PortfolioOptimizer

class VanillaBacktester:
    def __init__(self,price_path,split_date):
        self.split_date = split_date
        self.data = DataHandler(price_path)
        self.returns = self.data.compute_returns()
        self.strategy_returns = {}
        self.train_data,self.test_data = self.data.test_train_split(self.split_date)


    def compute(self):
        weights = self.get_weights(self.train_data)
        self.strategy_returns = self.compute_portfolio_returns(weights)
        return self.strategy_returns
    
    def compute_portfolio_returns(self, weights):
        portfolio_returns = {}
        for name, weight in weights.items():
            if not isinstance(weight, pd.Series): # So as to check if weight is numpy array or pandas array, if it is numpy array then numpy arrays are not named by colomuns whereas Pandas aligns by labels (index / column names), not by position. so AAPL will have certain weight and AAPL test_returns will have returns say they are in pos =3 in weights and pos=9 in test_returns still Pandas will multiply them coorectly as they are namned so this is extremely helpfull if later the order gets shuffled, numpy will always follow the matrix multiplication method as there column name is not present 
                weight = pd.Series(weight, index=self.test_data.columns)
            portfolio_returns[name] = self.test_data @ weight

        return portfolio_returns
    
    def get_weights(self,train_data):
        optimizer = PortfolioOptimizer(train_data)
        start_date = pd.to_datetime(train_data.index[0].strftime('%Y-%m-%d'))
        end_date = pd.to_datetime(self.split_date)
        # rand_weights = analyzer.random()
        self.risk_free_rate = self.data.get_risk_free_rate(start_date,end_date)
        return {
        "Tangency": optimizer.tangency(self.risk_free_rate ),
        "Min Variance": optimizer.min_variance(),
        "Equal Weight": optimizer.equal_weight(),
        "Random": optimizer.random()[0]
        }