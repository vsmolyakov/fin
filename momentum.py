
from zipline.api import order_target, order, record, symbol, history, add_history
import matplotlib.pyplot as plt

from datetime import datetime
import pytz

from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo
from zipline.utils.factory import load_bars_from_yahoo

import numpy as np


def initialize(context):
        
    #register 2 histories to track daily prices
    add_history(100, '1d', 'price')
    add_history(300, '1d', 'price')
    
    context.i = 0
    context.invested = False
    
def handle_data(context, data):
    
    #trading algorithm (executed on every event)
    
    #skip first 300 days to get full windows
    context.i += 1
    if context.i < 300:
        return
    
    #compute short and long moving averages:
    short_mavg = history(100, '1d', 'price').mean()
    long_mavg = history(300, '1d', 'price').mean()
    
    buy = False
    sell = False
    
    #trading logic
    if (short_mavg[0] > long_mavg[0]) and not context.invested:
        buy = True
        context.invested = True
        order_target(symbol('AAPL'), 100)        
    elif (short_mavg[0] < long_mavg[0]) and context.invested:
        sell = True
        context.invested = False
        order_target(symbol('AAPL'), -100)
    
    #save values for plotting
    record(AAPL = data[symbol('AAPL')].price,
           short_mavg = short_mavg[0],
           long_mavg = long_mavg[0],
           buy=buy,
           sell=sell)
       
                
def analyze(context=None, results=None, benchmark=None):
    
    hist_size = 300
        
    f, (ax1, ax2) = plt.subplots(2, sharex = True)        
    ax1.plot(results.portfolio_value[hist_size:], linewidth = 2.0, label = 'porfolio')
    ax1.set_title('Dual Moving Average Strategy')
    ax1.set_ylabel('Portfolio value (USD)')
    ax1.legend(loc=0)
    ax1.grid(True)
    
    ax2.plot(results['AAPL'][hist_size:], linewidth = 2.0, label = 'AAPL')
    ax2.plot(results['short_mavg'][hist_size:], color = 'r', linestyle = '-', linewidth = 2.0, label = 'short mavg')
    ax2.plot(results['long_mavg'][hist_size:], color = 'g', linestyle = '-', linewidth = 2.0, label = 'long mavg')
    ax2.set_ylabel('AAPL price (USD)')
    ax2.legend(loc=0)
    ax2.grid(True)

    plt.show()


if __name__ == "__main__":

    plt.close('all')    
            
    #load data
    #year, month, day, hour, minute, second, microsecond
    start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2015, 1, 1, 0, 0, 0, 0, pytz.utc)
    
    data = load_from_yahoo(stocks=['AAPL'], indexes={}, start=start, end=end)
    #data = load_bars_from_yahoo(stocks=['AAPL'], start=start, end=end)
    
    data.AAPL.tail(5) 
            
    #run the algorithm
    algo = TradingAlgorithm(initialize=initialize, handle_data = handle_data)    
    results = algo.run(data)
    
    #generate plots
    analyze(results = results)
        
    idx = np.where(results['buy'] == True)
    print "number of buys: %d" % np.size(idx[0])
    
    ##notes
    # see package requirements at:
    # etc/requirements.txt