import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

class RiskAnalyzer:
    def __init__(self,portfolio_returns):
        self.portfolio_returns = portfolio_returns

    def summary(self,alpha=0.95,target=0): # Here alpha is the confidence interval, Note it carefully. So we are confident the losses wont go more than VaR with alpha confidence
        risk_metrics = []
        self.var_historical(alpha)
        self.cvar_historical(alpha)
        self.var_parametric(alpha)
        self.cvar_parametric(alpha)
        self.downside_deviation(target)
        for name, var_hist in self.VaR_hist.items():
            risk_metrics.append({
            "Portfolio": name,
            "Var Historical": var_hist,
            "Cvar Historical": self.CVaR_hist[name],
            "Var Parametric": self.VaR_parametric[name],
            "Cvar Parametric": self.CVaR_parametric[name],
            "Downside Deviation": self.Downside_dev[name]
            })
        risk_metrics = pd.DataFrame(risk_metrics)
        return risk_metrics

    # Parametric VaR assumes the returns are normal
    def var_historical(self, alpha=0.95):
        self.VaR_hist = {}
        for name, returns in self.portfolio_returns.items():
            self.VaR_hist[name] = -np.percentile(returns , 100 * (1 - alpha)) 

    def var_parametric(self, alpha=0.95):
        #Parametric VaR may underestimate tail risk under non-normal returns.
        self.VaR_parametric = {}
        for name, returns in self.portfolio_returns.items():
                mean = returns.mean()
                std_dev = returns.std()
                z = norm.ppf(alpha) # Positive number
                self.VaR_parametric[name] = -(mean - z * std_dev)

    def cvar_historical(self, alpha=0.95):
        self.CVaR_hist = {}
        for name, returns in self.portfolio_returns.items():
            var_quantile = np.percentile(returns, 100 * (1 - alpha))
            self.CVaR_hist[name] = -returns[returns <= var_quantile].mean()

    def cvar_parametric(self, alpha=0.95):
        self.CVaR_parametric = {}
        for name, returns in self.portfolio_returns.items():
            mean = -returns.mean()
            std_dev = returns.std()
            z = norm.ppf(alpha)
            self.CVaR_parametric[name] = (mean + std_dev * norm.pdf(z) / (1 - alpha))

    def downside_deviation(self, target=0):
        self.Downside_dev = {}
        for name, returns in self.portfolio_returns.items():
            downside = np.maximum(0,  target-returns)
            self.Downside_dev[name] = np.sqrt((downside ** 2).mean())

    def var_cornish_fisher(self, alpha=0.95):
        # Cornish–Fisher VaR adjusts for skewness and kurtosis
        self.VaR_cornish_fisher = {}

        for name, returns in self.portfolio_returns.items():
            mean = returns.mean()
            std_dev = returns.std()

            skew = returns.skew()
            excess_kurt = returns.kurtosis()  
            z = norm.ppf(alpha)
            z_cf = (z+ (1/6) * (z**2 - 1) * skew+ (1/24) * (z**3 - 3*z) * excess_kurt- (1/36) * (2*z**3 - 5*z) * skew**2)
            self.VaR_cornish_fisher[name] = -(mean - z_cf * std_dev)
        return self.VaR_cornish_fisher


class RiskAnalyzerRolling:
    def __init__(self,portfolio_returns,step=1,window_size=252):
        self.window_size = window_size
        self.portfolio_returns = portfolio_returns
        self.step = step
        self.var_EWMA = None
        self.var_GARCH = None
        self.violations = None

    def var_parametric(self, alpha=0.95):
    #Parametric VaR may underestimate tail risk under non-normal returns.
        self.VaR_parametric = {}
        for name, returns in self.portfolio_returns.items():
            var_values = []
            dates = []
            for i in range(self.window_size, len(returns), self.step):
                window = returns.iloc[i - self.window_size : i]
                date   = returns.index[i]
                mean = window.mean()
                std_dev = window.std()
                z = norm.ppf(alpha) # positive number
                Var = (mean - z * std_dev)
                Var = -Var
                var_values.append(Var)
                dates.append(date)
            self.VaR_parametric[name] = pd.Series(var_values, index=dates)
        self.VaR_parametric = pd.DataFrame(self.VaR_parametric)
        return self.VaR_parametric 

    def var_historical(self, alpha=0.95):
        self.VaR_historical = {}
        for name, returns in self.portfolio_returns.items():
            var_values = []
            dates = []
            for i in range(self.window_size, len(returns), self.step):
                window = returns.iloc[i - self.window_size : i]
                date   = returns.index[i]
                Var = -np.percentile(window , 100 * (1 - alpha)) 
                var_values.append(Var)
                dates.append(date)
            self.VaR_historical[name] = pd.Series(var_values, index=dates)
        self.VaR_historical= pd.DataFrame(self.VaR_historical)
        return self.VaR_historical

    def cvar_parametric(self, alpha=0.95):
        self.CVaR_parametric = {}
        for name, returns in self.portfolio_returns.items():
                Cvar_values = []
                dates = []
                for i in range(self.window_size, len(returns), self.step):
                    window = returns.iloc[i - self.window_size : i]
                    date   = returns.index[i]
                    mean = -window.mean()
                    std_dev = window.std()
                    z = norm.ppf(alpha)
                    CVar = (mean + std_dev * norm.pdf(z) / (1 - alpha))
                    Cvar_values.append(CVar)
                    dates.append(date)
                self.CVaR_parametric[name] = pd.Series(Cvar_values, index=dates)
        self.CVaR_parametric= pd.DataFrame(self.CVaR_parametric)
        return self.CVaR_parametric

    def cvar_historical(self, alpha=0.95):
        self.Cvar_historical = {}
        for name, returns in self.portfolio_returns.items():
                Cvar_values = []
                dates = []
                for i in range(self.window_size, len(returns), self.step):
                    window = returns.iloc[i - self.window_size : i]
                    date   = returns.index[i]
                    var_quantile = np.percentile(window, 100 * (1 - alpha))
                    CVar = -returns[returns <= var_quantile].mean()
                    Cvar_values.append(CVar)
                    dates.append(date)
                self.Cvar_historical[name] = pd.Series(Cvar_values, index=dates)
        self.Cvar_historical= pd.DataFrame(self.Cvar_historical)
        return self.Cvar_historical

    def var_cornish_fisher(self, alpha=0.95):
        self.VaR_cornish_fisher = {}
        for name, returns in self.portfolio_returns.items():
                var_values = []
                dates = []
                for i in range(self.window_size, len(returns), self.step):
                    window = returns.iloc[i - self.window_size : i]
                    date   = returns.index[i]
                    mean = window.mean()
                    std_dev = window.std()
                    skew = window.skew()
                    excess_kurt = window.kurtosis()  
                    z = norm.ppf(alpha)
                    z_cf = (z+ (1/6) * (z**2 - 1) * skew+ (1/24) * (z**3 - 3*z) * excess_kurt- (1/36) * (2*z**3 - 5*z) * skew**2)
                    Var = -(mean - z_cf * std_dev)
                    var_values.append(Var)
                    dates.append(date)
                self.VaR_cornish_fisher[name] = pd.Series(var_values, index=dates)
        self.VaR_cornish_fisher= pd.DataFrame(self.VaR_cornish_fisher)
        return self.VaR_cornish_fisher
    def var_ewma(self, alpha=0.95,lambda_=0.94):
        self.var_EWMA = {}
        z = norm.ppf(alpha)
        for name, returns in self.portfolio_returns.items():
            var_EWMA_values = []
            dates = []
            initial_window = returns.iloc[:self.window_size] # Take elements from index 0 up to (but not including) self.window_size
            sigma2 = initial_window.var()
            for i in range(self.window_size, len(returns), self.step):
                r_prev = returns.iloc[i-1]
                sigma2 = lambda_ * sigma2 + (1 - lambda_) * (r_prev ** 2)
                sigma = np.sqrt(sigma2)
                VaR_t = z * sigma
                var_EWMA_values.append(VaR_t)
                dates.append(returns.index[i])

            self.var_EWMA[name] = pd.Series(var_EWMA_values, index=dates)
        self.var_EWMA= pd.DataFrame(self.var_EWMA)

        return self.var_EWMA

    def var_garch(self, alpha=0.95):
        from arch import arch_model
        self.var_GARCH = {}
        z = norm.ppf(alpha)
        for name, returns in self.portfolio_returns.items():
            var_GARCH_values = []
            dates = []
            for i in range(self.window_size, len(returns), self.step):
                window = returns.iloc[i - self.window_size : i]
                window_scaled = window * 100
                model = arch_model(
                window_scaled,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal")
                res = model.fit(disp="off")
                forecast = res.forecast(horizon=1)
                sigma2 = forecast.variance.iloc[-1, 0]
                sigma = np.sqrt(sigma2) / 100  # rescale back
                VaR_t = z * sigma
                var_GARCH_values.append(VaR_t)
                dates.append(returns.index[i])
            self.var_GARCH[name] = pd.Series(var_GARCH_values, index=dates)
        self.var_GARCH= pd.DataFrame(self.var_GARCH)
        return self.var_GARCH
    
    def backtest_var_violations(self, VaR_df):
        self.violations = {}
        for name, returns in self.portfolio_returns.items():
                violations = []
                dates = []
                for date in VaR_df.index:
                    if date not in returns.index:
                        continue
                    VaR_t = VaR_df.loc[date, name]
                    var = -VaR_t
                    ret = returns.loc[date]
                    if(ret<var):
                        violations.append(1)
                    else:
                        violations.append(0)
                    dates.append(date)
                self.violations[name] = pd.Series(violations, index=dates)
        self.violations= pd.DataFrame(self.violations)
        return self.violations

    def kupiec_test(self):
        self.kupiec_result = {}
        for name, violation in self.violations.items():
            n = violation.count()
            m = (violation == 1).sum()
            p = self.expected_probs[name]     
            p_hat = m / n                      
            eps = 1e-10
            p = np.clip(p, eps, 1 - eps)
            p_hat = np.clip(p_hat, eps, 1 - eps)
            l0 = (n - m) * np.log(1 - p) + m * np.log(p)
            l1 = (n - m) * np.log(1 - p_hat) + m * np.log(p_hat)
            self.kupiec_result[name] = -2 * (l0 - l1)

    def christoffersen_test(self):
        self.christoffersen_independence = {}
        self.christoffersen_uncond_cov = {}
        self.christoffersen_indep_and_uncond = {}

        for name, violation in self.violations.items():
            v = violation.values
            u00 = u01 = u10 = u11 = 0
            for i in range(1, len(v)):
                if v[i-1] == 0 and v[i] == 0:
                    u00 += 1
                elif v[i-1] == 0 and v[i] == 1:
                    u01 += 1
                elif v[i-1] == 1 and v[i] == 0:
                    u10 += 1
                elif v[i-1] == 1 and v[i] == 1:
                    u11 += 1

            eps = 1e-10
            pi01 = u01 / (u00 + u01 + eps)
            pi11 = u11 / (u10 + u11 + eps)
            pi   = (u01 + u11) / (u00 + u01 + u10 + u11 + eps)
            pi01 = np.clip(pi01, eps, 1 - eps)
            pi11 = np.clip(pi11, eps, 1 - eps)
            pi   = np.clip(pi, eps, 1 - eps)
            ll_indep = (
                u00 * np.log(1 - pi01) +
                u01 * np.log(pi01) +
                u10 * np.log(1 - pi11) +
                u11 * np.log(pi11)
            )

            ll_null = (
                (u00 + u10) * np.log(1 - pi) +
                (u01 + u11) * np.log(pi)
            )
            LR_independence = -2 * (ll_null - ll_indep)
            LR_cc = self.kupiec_result[name] + LR_independence
            self.christoffersen_independence[name] = LR_independence
            self.christoffersen_uncond_cov[name] = self.kupiec_result[name]
            self.christoffersen_indep_and_uncond[name] = LR_cc



    def violation_summary(self, alpha=0.95):
        if self.violations is None or self.violations.empty:
            raise RuntimeError("No violations to analyze")
        self.true_probs = {}
        self.expected_probs = {}
        for name, violation in self.violations.items():
            self.true_probs[name] = violation.mean()
            self.expected_probs[name] = 1 - alpha
        self.kupiec_test()
        self.christoffersen_test()
        summary_dict = {}
        for name in self.violations.columns:
            kupiec_lr = self.kupiec_result[name]
            p_kupiec = chi2.sf(kupiec_lr, 1)
            
            ind_lr = self.christoffersen_independence[name]
            p_ind = chi2.sf(ind_lr, 1)

            cc_lr = self.christoffersen_indep_and_uncond[name]
            p_cc = chi2.sf(cc_lr, 2)

            summary_dict[name] = {
                "Expected violation prob": self.expected_probs[name],
                "Observed violation prob": self.true_probs[name],

                "Kupiec LR": kupiec_lr,
                "Probability Kupiec": p_kupiec,
                "Kupiec Result":"PASS" if p_kupiec > 0.05 else "FAIL",

                "Christoffersen Independence LR": ind_lr,
                "Probability Christoffersen Independence": p_ind,
                "Christoffersen Independence Result":"PASS" if p_ind > 0.05 else "FAIL",

                "Christoffersen Conditional LR": cc_lr,
                "Probability Christoffersen Conditional": p_cc,
                "Christoffersen Conditional Result": "PASS" if p_cc > 0.05 else "FAIL"
            }

        self.summary = pd.DataFrame(summary_dict)
        print(self.summary)
        return self.summary
    
    def get_GARCH_EWMA_plot_data(self,name="Tangency"):
        if self.var_EWMA is None:
            self.var_ewma()
        if self.var_GARCH is None:
            self.var_garch()
        if self.violations is None:
            raise RuntimeError("Run backtest_var_violations first")
        viol = self.violations[name]
        violation_dates = viol[viol == 1].index
        return {
            "returns": self.portfolio_returns[name],
            "violation_dates": violation_dates,
            "garch": -self.var_GARCH[name],
            "ewma": -self.var_EWMA[name],
        }