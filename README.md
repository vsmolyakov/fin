# fin
Algorithmic Trading

### Description

**Gaussian Process Regression**

Gaussian Proccess (GP) is a way to perform Bayesian inference over functions. A GP assumes that p(f(x1),...,f(xn)) is jointly Gaussian with mean mu(x) and covariance Sigma(x) given by Sigma_ij = k(xi,xj), where k is a positive definite kernel.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/market_gp.png"/>
</p>

The figure above shows GP regression applies to SP500 time series for a period of one year. Notice that the lack of future observations results in a constant prediction, i.e. the mean and the variance of the GP regressor do not change.


**Mean-Variance Portfolio**

The objective of mean-variance analysis is to maximize the expected return of a portfolio for a given level of risk as measured by the standard deviation of past returns. By varying the mixing proportions of each asset we can achieve different risk-return trade-offs.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/portfolio_merged.png"/>
</p>

The figure above shows regression results between pairs of portfolio assets. Notice, for example, how SPY is uncorrelated with TIP and anti-correlated with GLD. Also, the diagonal densities are multi-modal and show negative skewness for riskier assets (e.g. SPY vs LQD). The figure in the top right, shows the expected return vs risk trade-off for a set of randomly generated portfolios. The efficient frontier is defined by a set of portfolios at the top of the curve that correspond to maximum expected return for a given standard deviation. By adding a risk free asset, we can choose a portfolio along a tangent line with slope defined by the Sharpe ratio.


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