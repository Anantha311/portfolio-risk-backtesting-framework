import pandas as pd
from data.data_handler import DataHandler
from optimization.portfolio_optimizer import PortfolioOptimizer
import numpy as np

class RollingBacktester:
    def __init__(self, price_path,tran_cost=True,method_tran = "Turnover",train_period=6,rollback_period=3): # Give the entire returns, and roll back period is in months
        self.prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        self.tran_cost = tran_cost
        self.method_tran = method_tran
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
        weights_new = None
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
                weights_old = weights_new
                weights_new = self.get_weights(train_data,start_date=train_start,end_date=train_end)
                temp_portfolio_returns = self.compute_portfolio_returns(test_data,weights_new)
                if self.tran_cost and weights_old is not None:
                    if self.method_tran == "Volatility":
                        vol = self.cal_volatility(train_data,weights_old)
                        temp_portfolio_returns = self.transaction_cost(temp_portfolio_returns,self.method_tran,vol=vol,test_start=test_start,weight_old=weights_old,weight_new=weights_new)
                    elif self.method_tran == "Liquidity":
                        adv = self.cal_adv(train_start,train_end)
                        vol_asset = train_data.std()
                        temp_portfolio_returns = self.transaction_cost(temp_portfolio_returns,self.method_tran,vol_asset=vol_asset,adv=adv,test_start=test_start,weight_old=weights_old,weight_new=weights_new)
                    else:
                        temp_portfolio_returns = self.transaction_cost(temp_portfolio_returns,self.method_tran,test_start=test_start,weight_old=weights_old,weight_new=weights_new)
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
    
    def transaction_cost(self,portfolio_returns,method,test_start,vol=None,adv=None,vol_asset=None,weight_old=None,weight_new =None,turnover_cost_rate=0.001,volatility_cost_rate=0.001,liquidity_cost_rate=0.001):
        if method == "Turnover" and weight_new is not None and weight_old is not None:
            portfolio_returns = self.turnover_based_cost(portfolio_returns,weight_old,weight_new,turnover_cost_rate)
        elif method == "Volatility" and weight_new is not None and weight_old is not None:
            portfolio_returns = self.volatility_based_cost(portfolio_returns, weight_old, weight_new,volatility_cost_rate,vol=vol)
        elif method == "Spread" and weight_new is not None and weight_old is not None:
            portfolio_returns = self.spread_based_cost(portfolio_returns, weight_old, weight_new,test_start)
        elif method == "Liquidity" and weight_new is not None and weight_old is not None:
            portfolio_returns = self.liquidity_based_cost(portfolio_returns,weight_old,weight_new,adv,vol_asset,liquidity_cost_rate)

        return portfolio_returns

    def turnover_based_cost(self, portfolio_returns, weight_old, weight_new,cost_rate):
        new_portfolio_return = {}
        for name, returns in portfolio_returns.items():
            turnover = np.sum(np.abs(weight_old[name] - weight_new[name]))
            cost = cost_rate * turnover  
            returns = returns.copy()
            returns.iloc[0] -= cost   
            new_portfolio_return[name] = returns
        return new_portfolio_return
    
    def volatility_based_cost(self, portfolio_returns, weight_old, weight_new,cost_rate,vol):
        new_portfolio_return = {}
        for name, returns in portfolio_returns.items():
            turnover = np.sum(np.abs(weight_old[name] - weight_new[name]))
            sigma = vol[name] * np.sqrt(252)
            cost = cost_rate * turnover * sigma
            returns = returns.copy() # without .copy() I might unintentionally modify the original data, causing silent bugs and incorrect backtest results.
            returns.iloc[0] -= cost   
            new_portfolio_return[name] = returns
        return new_portfolio_return
    
    def cal_volatility(self, train_data, weights):
        vol = {}
        for name, w in weights.items():
            if not isinstance(w, pd.Series):
                w = pd.Series(w, index=train_data.columns)
            port_ret = train_data @ w
            vol[name] = port_ret.std()
        return vol

    def spread_based_cost(self, portfolio_returns, weight_old, weight_new,test_start):
        print(test_start)
        spread_proxy = pd.read_csv("data/spread_proxy.csv", index_col=0, parse_dates=True)
        spread_t = spread_proxy.loc[:test_start].iloc[-1]
        new_portfolio_return = {}
        for name, returns in portfolio_returns.items():
            turnover = np.sum(np.abs(weight_old[name] - weight_new[name]))
            weighted_spread = np.sum(weight_new[name] * spread_t)
            cost = 0.5 * turnover *  weighted_spread
            returns = returns.copy()
            returns.iloc[0] -= cost   
            new_portfolio_return[name] = returns
        return new_portfolio_return
    
    def cal_adv(self, train_start, train_end, window=20):
        adv_data = pd.read_csv("data/volume.csv", index_col=0, parse_dates=True)
        volume = adv_data.loc[train_start:train_end]
        adv = volume.tail(window).mean()
        return adv
    
    def liquidity_based_cost(self,portfolio_returns,weight_old,weight_new,adv,asset_vol,cost_rate,eps=1e-8):
        new_portfolio_return = {}
        for name, returns in portfolio_returns.items():
            dw = np.abs(weight_new[name] - weight_old[name])
            dw = dw.reindex(asset_vol.index)
            cost = cost_rate * (dw * asset_vol / adv).sum()
            r = returns.copy()
            r.iloc[0] -= cost
            new_portfolio_return[name] = r
        return new_portfolio_return
    
    def non_linear_based_cost(self, portfolio_returns, weight_old, weight_new,k):
        new_portfolio_return = {}
        for name, returns in portfolio_returns.items():
            turnover = np.sum(np.abs(weight_old[name] - weight_new[name]))
            cost = k * (turnover**1.5)  
            returns = returns.copy()
            returns.iloc[0] -= cost   
            new_portfolio_return[name] = returns
        return new_portfolio_return
