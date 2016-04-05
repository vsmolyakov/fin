# fin
Algorithmic Trading

### Description

**Momentum**

The momentum strategy is based on the difference between short and long term trends.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/momentum.png" width = "400"/>
</p>

The figure above shows short and long term averages computed for AAPL. A buy signal is issued when the short-term trend crosses the long-term trend from below indicated an upward momentum.



### Dependencies

Python 2.7  
Sklearn 0.17  
Pandas 0.16.1  
Zipline 0.7.0  