import pandas as pd
import numpy as np

class PerformanceMetrics:
    @staticmethod
    def annualized_return(returns, periods_per_year=252): # Here returns is test_returns @ weights (will be of shape (T,) where T is number of days)
        compounded = (1 + returns).prod() # .prod() means product of all elements:
        n_years = len(returns) / periods_per_year
        return compounded ** (1 / n_years) - 1

    @staticmethod
    def annualized_volatility(returns, periods_per_year=252):
        return returns.std() * np.sqrt(periods_per_year) # we multiply by (252)**1/2 to convert std_daily to std_yearly

    @staticmethod
    def sharpe_ratio(returns, rf, periods_per_year=252):
        excess = (returns - (rf / periods_per_year)) # Converts annual risk-free rate to per-period
        return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def max_drawdown(equity_curve):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def sortino_ratio(returns, rf, target=0, periods_per_year=252):
        excess = returns - rf / periods_per_year
        downside = np.maximum(0, target - returns)
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
        return excess.mean() * periods_per_year / downside_dev
    
    def compute(portfolio_returns, equity_curve,rf):
        performence_metrics = []
        for name, returns in portfolio_returns.items():
            performence_metrics.append(
            {
            "Portfolio": name,
            "Annualized Return": PerformanceMetrics.annualized_return(returns),
            "Annualized Volatility": PerformanceMetrics.annualized_volatility(returns),
            "Sharpe Ratio": PerformanceMetrics.sharpe_ratio(returns, rf),
            "Sortino Ratio": PerformanceMetrics.sortino_ratio(returns, rf),
            "Max Drawdown": PerformanceMetrics.max_drawdown(equity_curve[name])
            }
            )
        performence_metrics = pd.DataFrame(performence_metrics)
        return performence_metrics


class StressPerformenceMetrics:
    @staticmethod
    def annualized_return(returns, periods_per_year=252): # Here returns is test_returns @ weights (will be of shape (T,) where T is number of days)
        compounded = (1 + returns).prod() # .prod() means product of all elements:
        n_years = len(returns) / periods_per_year
        return compounded ** (1 / n_years) - 1

    @staticmethod
    def annualized_volatility(returns, periods_per_year=252):
        return returns.std() * np.sqrt(periods_per_year) # we multiply by (252)**1/2 to convert std_daily to std_yearly

    @staticmethod
    def max_drawdown(equity_curve):
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    def compute(portfolio_returns, equity_curve):
        stress_performence_metrics = []
        for name, returns in portfolio_returns.items():
            stress_performence_metrics.append(
            {
            "Portfolio": name,
            "Annualized Return": StressPerformenceMetrics.annualized_return(returns),
            "Annualized Volatility": StressPerformenceMetrics.annualized_volatility(returns),
            "Max Drawdown": StressPerformenceMetrics.max_drawdown(equity_curve[name])
            }
            )
        stress_performence_metrics = pd.DataFrame(stress_performence_metrics)
        return stress_performence_metrics
