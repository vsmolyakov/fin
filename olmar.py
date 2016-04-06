
from zipline.api import order, record, symbol, history, add_history
import matplotlib.pyplot as plt

from datetime import datetime
import pytz

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo
from zipline.finance import commission, slippage

import numpy as np

import pdb

STOCKS = ['AAPL', 'MSFT']
SIDS = [symbol('AAPL'), symbol('MSFT')]

def initialize(context, eps = 10, window_length = 50):
    
    #init    
    context.stocks = STOCKS
    context.sids = SIDS
    #context.sids = [context.symbol(symb) for symb in context.stocks]
    context.m = np.size(STOCKS)
    context.price = {}
    context.b_t = np.ones(context.m)/float(context.m)
    context.prev_weights = np.ones(context.m)/float(context.m)
    context.eps = eps
    context.init = True
    context.days = 0
    context.window_length = window_length
    
    add_history(window_length, '1d', 'price')
    
    #set commision and slippage
    #context.set_commision(commission.PerShare(cost=0))
    #context.set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))    
    
    
def handle_data(context, data):
    
    #On-Line Moving Average Reversal (OLMAR)
    
    context.days += 1
    if context.days < context.window_length:
        return
    
    if context.init:
        rebalance_portfolio(context, data, context.b_t)
        context.init=False
        return
    
    m = context.m            #num assets
    x_tilde = np.zeros(m)    #relative mean deviation
    b = np.zeros(m)          #weights
    
    #compute moving average price for each asset
    mavgs = history(context.window_length, '1d', 'price').mean()    
    #mavgs = data.history(context.sids, 'price', context.window_length, '1d').mean()
    
    for i, stock in enumerate(context.stocks):
        price = data[stock]['price']
        x_tilde[i] = mavgs[i] / price
    
    x_bar = x_tilde.mean()
    
    market_rel_dev = x_tilde - x_bar  #relative deviation
    
    exp_return = np.dot(context.b_t, x_tilde)
    weight = context.eps - exp_return
    variability = (np.linalg.norm(market_rel_dev))**2
    
    if variability == 0.0:
        step_size = 0
    else:
        step_size = np.max((0, weight/variability))
    
    
    b = context.b_t + step_size * market_rel_dev
    b_norm = simplex_projection(b)
    
    rebalance_portfolio(context, data, b_norm)

    context.b_t = b_norm
                    
    #save values for plotting
    record(AAPL = data[symbol('AAPL')].price,
           MSFT = data[symbol('MSFT')].price,
           step_size = step_size,
           variability = variability
           )

def rebalance_portfolio(context, data, weights):    
    
    desired_amount = np.zeros(np.shape(weights))
    current_amount = np.zeros(np.shape(weights))        
    prices = np.zeros(np.shape(weights))

    if context.init:
        positions_value = context.portfolio.starting_cash
    else:
        #total cash
        positions_value = context.portfolio.positions_value + context.portfolio.cash
    
    for i, stock in enumerate(context.stocks):
        current_amount[i] = context.portfolio.positions[stock].amount  #shares
        prices[i] = data[stock]['price'] #share price            

    context.prev_weights = weights
    desired_amount = np.round(weights * positions_value / prices)    #shares    
    diff_amount = desired_amount - current_amount

    #pdb.set_trace()        
    for i, sid in enumerate(context.sids):
        order(sid, +diff_amount[i])

def simplex_projection(v, b=1):
    
    v = np.array(v)
    p = np.size(v)
    
    v = (v > 0)*v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    
    rho = np.where(u > (sv-b) / np.arange(1,p+1))[0][-1]
    theta = np.max([0, (sv[rho]-b)/(rho+1)])
    w = v - theta
    w[w<0] = 0
    
    return w
                
                                                
def analyze(context=None, results=None):
        
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True)        
    ax1.plot(results.portfolio_value, linewidth = 2.0, label = 'porfolio')
    ax1.set_title('On-Line Moving Average Reversion')
    ax1.set_ylabel('Portfolio value (USD)')
    ax1.legend(loc=0)
    ax1.grid(True)
            
    ax2.plot(results['AAPL'], color = 'b', linestyle = '-', linewidth = 2.0, label = 'AAPL')
    ax2.plot(results['MSFT'], color = 'r', linestyle = '-', linewidth = 2.0, label = 'MSFT')
    ax2.set_ylabel('stock price (USD)')
    ax2.legend(loc=0)
    ax2.grid(True)
    
    ax3.semilogy(results['step_size'], color = 'b', linestyle = '-', linewidth = 2.0, label = 'step-size')
    ax3.semilogy(results['variability'], color = 'r', linestyle = '-', linewidth = 2.0, label = 'variability')
    ax3.legend(loc=0)
    ax3.grid(True)
    
    plt.show()


if __name__ == "__main__":

    plt.close('all')    
            
    #load data
    #year, month, day, hour, minute, second, microsecond
    start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)    

    data = load_from_yahoo(stocks=STOCKS, indexes={}, start=start, end=end)
    data = data.dropna()
            
    #run the algorithm
    olmar = TradingAlgorithm(initialize=initialize, handle_data = handle_data)    
    results = olmar.run(data)
    
    #generate plots
    analyze(results = results)
        
