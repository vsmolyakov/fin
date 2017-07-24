import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_datareader import data

import pymc3 as pm

np.random.seed(0)

def main():

    #load data    
    returns = data.get_data_google('SPY', start='2008-5-1', end='2009-12-1')['Close'].pct_change()
    returns.plot()
    plt.ylabel('daily returns in %');
    
    with pm.Model() as sp500_model:
        
        nu = pm.Exponential('nu', 1./10, testval=5.0)
        sigma = pm.Exponential('sigma', 1./0.02, testval=0.1)
        
        s = pm.GaussianRandomWalk('s', sigma**-2, shape=len(returns))                
        r = pm.StudentT('r', nu, lam=pm.math.exp(-2*s), observed=returns)
        
    
    with sp500_model:
        trace = pm.sample(2000)

    pm.traceplot(trace, [nu, sigma]);
    plt.show()
    
    plt.figure()
    returns.plot()
    plt.plot(returns.index, np.exp(trace['s',::5].T), 'r', alpha=.03)
    plt.legend(['S&P500', 'stochastic volatility process'])
    plt.show()

    
if __name__ == "__main__":
    main()