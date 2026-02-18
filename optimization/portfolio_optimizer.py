import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet


class PortfolioOptimizer:
    def __init__(self, returns):
        self.mu = returns.mean() * 252
        self.cov = returns.cov() * 252
        self.n = len(self.mu)
     
    def equal_weight(self):
        return np.ones(self.n)/self.n
    
    def min_variance(self):
        def objective_function(x):
            return x @ self.cov @ x
        def equality_constraint(x):
            return np.sum(x) - 1
        bounds = tuple((0, 1) for _ in range(self.n))
        constraints = [{'type': 'eq', 'fun': equality_constraint}]
        intial_guess = self.equal_weight()
        result = minimize(objective_function,intial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
        if not result.success:
            raise RuntimeError("Min-variance optimization failed")
        return result.x
    
    def tangency(self, rf):
        def objective_function(x):
            return -(((self.mu @ x) - rf)/(np.sqrt(x @ self.cov @ x)))
        def equality_constraint_1(x):
            return np.sum(x) - 1
        bounds = tuple((0, 1) for _ in range(self.n))
        constraints = [
        {'type': 'eq', 'fun': equality_constraint_1}]
        intial_guess = np.ones(self.n)/self.n
        result = minimize(objective_function,intial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
        if not result.success:
            raise RuntimeError("Tangency optimization failed")
        return result.x
    
    def min_var_for_given_return(self,given_return):
        def objective_function(x):
            return x @ self.cov @ x
        def equality_constraint_1(x):
            return np.sum(x) - 1
        def equality_constraint_2(x):
            return self.mu @ x - given_return
        bounds = tuple((0, 1) for _ in range(self.n))
        constraints = [{'type': 'eq', 'fun': equality_constraint_1}, {'type': 'eq', 'fun': equality_constraint_2}]
        intial_guess = np.ones(self.n)/self.n 
        result = minimize(objective_function,intial_guess,method='SLSQP',bounds=bounds,constraints=constraints)
        if not result.success:
            return None
        return result.x 

    def random(self,number_of_random_portfolios= 500):
        rand_std_dev = []
        rand_ret = []
        weights = []
        for _ in range(number_of_random_portfolios):
            w = dirichlet.rvs(np.ones(self.n))[0]
            weights.append(w)
            rand_std_dev.append(float(np.sqrt(w @ self.cov @ w)))
            rand_ret.append(float(self.mu @ w))
        rand_std_dev,rand_ret,weights = np.array(rand_std_dev),np.array(rand_ret),np.array(weights)
        return  weights
    
