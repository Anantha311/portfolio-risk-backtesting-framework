from analysis.analyzer import Analyzer
from analysis.performance import StressPerformenceMetrics

class StressTester:
    def __init__(self, portfolio_returns, alpha=0.95):
        self.portfolio_returns = portfolio_returns

    def historical_scenario(self,start_date,end_date,name_scenario="None"):
        stress_portfolio_returns = {} 
        for name, returns in self.portfolio_returns.items():
            stress_portfolio_returns[name] = returns[start_date:end_date]
        cumulative_returns = Analyzer.cumulative_returns(stress_portfolio_returns)
        Analyzer.plot_dataframe(curves=cumulative_returns ,title= f"Historical stress {name_scenario} -> Cumulative")
        Analyzer.plot_drawdowns(cumulative_returns,title=f"Historical stress {name_scenario} -> Drawdown")
        performence_metrics= StressPerformenceMetrics.compute(stress_portfolio_returns ,cumulative_returns )
        print(performence_metrics)
        
    def hypothetical_scenario(self,shock_type="equity",magnitude=-0.30,name="Equity Crash",):
        pass
    
