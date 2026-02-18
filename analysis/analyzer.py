import matplotlib.pyplot as plt
import numpy as np

class Analyzer:
    def __init__(self,optimizer):
        self.optimizer = optimizer
    
    @staticmethod
    def drawdown(cumulative_returns): 
        peak = cumulative_returns.cummax() #cummax() computes the running maximum of the portfolio value. So at each date: it remembers the highest value ever achieved up to that point
        drawdown = (cumulative_returns - peak) / peak 
        return drawdown
    
    @staticmethod
    def plot_dataframe(curves, title):
        plt.figure(figsize=(10, 5)) 
        for name, curve in curves.items():
            plt.plot(curve, label=name)
        plt.legend()
        plt.title(title)


    @staticmethod
    def plot_drawdowns(curves,title):
        plt.figure(figsize=(10, 5)) 
        for name, curve in curves.items():
            plt.plot(Analyzer.drawdown(curve), label=name)
        plt.legend()
        plt.title(title)

    def plot_garch_ewma_var(data, title="GARCH vs EWMA VaR"):
        plt.figure(figsize=(10, 5))
        plt.plot(data["returns"], color="gray", alpha=0.35, label="Returns")
        plt.scatter(
            data["violation_dates"],
            data["returns"].loc[data["violation_dates"]],
            marker="o",
            label="VaR Violations"
        )
        plt.plot(data["garch"], linewidth=2.2, label="GARCH VaR")
        plt.plot(data["ewma"], linewidth=2.2, label="EWMA VaR")
        plt.legend()
        plt.title(title)

    @staticmethod
    def plot_cml_random_efficient(ef_std_dev, ef_ret,rand_std_dev, rand_rets,cml_std_dev, cml_rets,tan_std_dev, tan_ret,eq_std_dev,eq_ret,min_var_std_dev,min_var_ret):
        plt.figure(figsize=(10, 6))
        plt.plot(ef_std_dev , ef_ret , label="Efficient Frontier")
        plt.scatter(rand_std_dev,rand_rets,alpha=0.1,label="Random Portfolios")
        plt.plot(cml_std_dev , cml_rets , label="Capital Market Line")
        plt.scatter(eq_std_dev,eq_ret,marker='s',s=100,label="Equal Weight Portfolio")
        plt.scatter(min_var_std_dev, min_var_ret, s=100, marker='*', label="Min Variance")
        plt.scatter(tan_std_dev, tan_ret, s=100, marker='D', label="Tangency")
        plt.xlabel("x")
        plt.ylabel("y")


    @staticmethod
    def cumulative_returns(portfolio_returns):
        cumulative_returns = {}
        for name, ret in portfolio_returns.items():
            cumulative_returns[name] = (1 + ret).cumprod()  #cumulative_returns is last returns say it is 1.02 it means your 1 became 1.02 a 2% increase, cumprod() does cumulative product and (1 + portfolio_returns) is to convert  portfolio_returns which are for e.g 0.02 (2%) to 1.02
        return cumulative_returns
    

    def effecient_frontier(self,number_of_points=30):
        minimum_variance_weights = self.optimizer.min_variance()
        min_var_return = float(self.optimizer.mu @ minimum_variance_weights )
        target_returns = np.linspace(min_var_return, self.optimizer.mu.max(), number_of_points)
        ef_std_dev = []
        ef_ret = []
        weights = []
        for return_i in target_returns:
            temp_weights = self.optimizer.min_var_for_given_return(given_return = return_i)
            temp_return,temp_std_dev = float(self.optimizer.mu @temp_weights), float(np.sqrt(temp_weights@self.optimizer.cov@temp_weights))
            if temp_weights is None:
                continue
            ef_std_dev.append(temp_std_dev)
            ef_ret.append(temp_return)
            weights.append(temp_weights)
        
        ef_std_dev,ef_ret,weights = np.array(ef_std_dev),np.array(ef_ret),np.array(weights)
        return ef_std_dev,ef_ret,weights

    def cml(self,rf,number_of_points=30,maximum_std_dev= 0.5):
        tangency_weights= self.optimizer.tangency(rf)
        tangency_return, tangency_std_dev = self.optimizer.mu @ tangency_weights, np.sqrt(tangency_weights @ self.optimizer.cov @ tangency_weights)
        cml_std_dev = np.linspace(0, maximum_std_dev, number_of_points)
        tangecy_slope = (((tangency_return)-rf)/(tangency_std_dev))
        cml_returns = []
        for std_dev in cml_std_dev:
            return_i = rf + tangecy_slope*std_dev
            cml_returns.append(return_i)
        cml_std_dev,cml_ret = np.array(cml_std_dev),np.array(cml_returns)
        return cml_std_dev, cml_ret
