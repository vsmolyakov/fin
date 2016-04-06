# fin
Algorithmic Trading

### Description

**On-Line Portfolio Selection with Moving Average Reversion**

On-Line Moving Average Reversion is an algorithm for optimum portfolio allocation. It assumes that any changes in stock price will revert to its moving average for a given reversion treshold epsilon and window size w.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/olmar.png"/>
</p>

The figure above shows the resulting portfolio value consisting of AAPL and MSFT stocks weighted by the algorithm. The bottom figure shows variability and step size used in updating portfolio proportions. It turns out the algorithm is sensitive to the moving average window size. The sensitivity can be improved by averaging portfolio weights for different window sizes weighted by historical performance.

References:  
*B. Li, "On-Line Portfolio Selection with Moving Average Reversion", ICML 2012*

**Momentum**

The momentum strategy is based on the difference between short and long term trends. The figure below shows short and long term averages computed for AAPL.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/momentum.png"/>
</p>

A buy signal is issued when the short-term trend crosses the long-term trend from below, indicating an upward momentum.


### Dependencies

Python 2.7  
Sklearn 0.17  
Pandas 0.16.1  
Zipline 0.7.0  