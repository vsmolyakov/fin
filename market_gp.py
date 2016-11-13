import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data, wb

from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split

from datetime import datetime
import pytz

np.random.seed(0)

def f(x): return x * np.sin(x)

if __name__ == "__main__":
   
   
    plt.close('all')
   
    #example: fit a GP (with noisy observations)
    X = np.array([1., 3., 5., 6., 7., 8.]).reshape(-1,1)
    y = f(X).ravel()
    dy = 0.5 + 1.0*np.random.random(y.shape)  #in [0.5, 1.5] <- std deviation per point
    y = y + np.random.normal(0, dy)  #0-mean noise with variable std in [0.5, 1.5]
    gp = GaussianProcess(corr='cubic', nugget = (dy / y)**2, theta0=1e-1, thetaL=1e-3, thetaU=1, random_start=100, verbose=True)
    gp.fit(X, y)  #ML est
    gp.get_params()
        
    Xt = np.array(np.linspace(np.min(X)-10,np.max(X)+10,1000)).reshape(-1,1)
    y_pred, MSE = gp.predict(Xt, eval_MSE=True)
    sigma = np.sqrt(MSE)
    
    plt.figure()
    plt.plot(Xt, f(Xt), color='k', lw=2.0, label = 'x sin(x) ground truth')
    plt.plot(X, y, 'r+', markersize=20, lw=2.0, label = 'observations')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')    
    plt.plot(Xt, y_pred, color = 'g', linestyle = '--', lw=1.5, label = 'GP prediction')
    plt.fill(np.concatenate([Xt, Xt[::-1]]), np.concatenate([y_pred-1.96*sigma, (y_pred+1.96*sigma)[::-1]]), alpha = 0.5, label = '95% conf interval')
    plt.title('GP regression')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.legend()
    plt.show()
            
    #fit a GP to market data
    #load data     
    start = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)    
    spy = data.DataReader("SPY", 'google', start, end)
    
    spy_price = np.array(spy['Close'].values).reshape(-1,1)
    spy_volume = np.array(spy['Volume'].values).reshape(-1,1)
    spy_obs = np.hstack([spy_price, spy_volume])
                
    #X = np.random.rand(np.size(spy_price)).reshape(-1,1)        
    X = np.array(range(np.size(spy_price))).reshape(-1,1)
    y = spy_price.ravel()
    dy = 10*spy.std()['Close']
    spy_gp = GaussianProcess(corr='cubic', nugget = (dy/y)**2, theta0=1e-1, thetaL=1e-3, thetaU=1e3, random_start=100, verbose=True)
    spy_gp.fit(X,y)
    
    spy_gp.get_params()
        
    Xt = np.array(np.linspace(np.min(X)-10,np.max(X)+10,1000)).reshape(-1,1)
    y_pred, MSE = spy_gp.predict(Xt, eval_MSE=True)
    sigma = np.sqrt(MSE)            

    f = plt.figure()
    plt.plot(X, y, 'r-', markersize=20, lw=2.0, label = 'SPY price, USD')
    #plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')    
    plt.plot(Xt, y_pred, color = 'g', linestyle = '--', lw=1.5, label = 'GP prediction')
    plt.fill(np.concatenate([Xt, Xt[::-1]]), np.concatenate([y_pred-1.96*sigma, (y_pred+1.96*sigma)[::-1]]), alpha = 0.5, label = '95% conf interval')
    plt.title('GP regression')
    plt.xlabel('time, days')
    plt.ylabel('S&P500 price, USD')
    plt.grid(True)
    plt.legend()
    plt.show()
    #f.savefig('./figures/market_gp.png')
                                        
    #notes:
    #sensitive to corr type, e.g. corr='cubic' is important
    # regr = 'linear' parameter
    # n_features > 1, i.e. price and volume   
    #http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_regression.html
    #http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcess.html
    #TODO: compare with other indicators (e.g. moving average filter)