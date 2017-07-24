# fin
Algorithmic Trading

### Description

**Pairs Trading Strategy**

A pairs trading strategy consists of identifying similar pairs of stocks and taking a linear combination of their price so that the result is a stationary time series. We can then compute z-scores for the stationary signal and trade on the spread assuming mean reversion: short the top asset and long the bottom asset.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/pairs_trading.png"/>
</p>

The figure above shows spread z-scores for Coke and Pepsi stocks. When the z-score is outside the +/- 1 band, we bet it's going to mean revert. So we long the bottom asset when the zscore is less than -1 and we short the top asset when the zscore is greater than 1. And we clear positions inside the band.

References:  
*https://www.quantopian.com/lectures*


**Long Short Strategy**

A long short strategy consists of selecting a universe of equities or futures and ranking them according to a combined alpha factor. Given the rankings, we long the top percentile and short the bottom percentile of securities once every re-balancing period. 

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/long_short.png"/>
</p>

The figure above shows the top minus the bottom quantile 1 day forward return based on combined ranking of three factors: momentum, revenue growth and P/E ratio. We can see the difference in quantile mean return fluctuating around zero.

References:  
*https://www.quantopian.com/lectures*


**Alpha Factor Selection**

A successful trading strategy consists of selecting a universe of securities, alpha factor modeling, alpha combination, risk modeling, portfolio construction and execution. Alpha factors express a predictive relationship between a given set of information and future returns. Discovering informative alpha factors is one of the crucial parts of a trading strategy.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/alpha_selection.png"/>
</p>

Alpha factors can range from returns and pricing movements to fundamental, technical and sentiment indicators. We can use machine learning in order to select the most predictive alphas by training an ensemble model such as Adaboost Classifier. By using alpha factors as features we can obtain a ranking of the most informative features for a given prediction horizon. We can then rank our securities universe using the combined alpha factor as key and include the top K informative alphas in a trading strategy. In addition, we can use Quantopian AlphaLens tool to quantify the predictive ability of the combined alpha factor. 

References:  
*https://www.quantopian.com/posts/machine-learning-on-quantopian*

**Stochastic Volatility**

Asset returns have a time-varying volatility. We use a bayesian model to describe the time varying nature of volatility in which the returns are T-distributed with variance that follows a Gaussian Random Walk. PyMC3 is used to infer the latent volatility process.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/stoch_volatility.png"/>
</p>

The figure above shows the S&P500 returns and posterior volatility samples obtained with MCMC. On the left, we can see the time-varying posterior volatility and on the right, we have trace plots for the degrees of freedom of T-distributed returns and a prior on standard deviation for the Gaussian Random Walk.

References:  
*http://pymc-devs.github.io/pymc3/notebooks/stochastic_volatility.html*


**Recurrent Neural Network**

Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) was trained on the closing price of S&P500 time series data. With a lookback window of 10 days, the LSTM network was used to predict the future values of the S&P500 index.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/lstm.png"/>
</p>

The figure above shows LSTM predictions on the training (green) and the test (red) time series. We can see the regression results closely match the actual market price.

**Stock Clusters**

Inverse covariance (precision) estimation is useful in constructing a graph network of dependencies. Here, the difference between opening and closing daily prices was used to compute empirical covariance that was fit with graph lasso algorithm to estimate sparse precision matrix. Affinity propagation was used to compute the clusters of stocks and a linear embedding was used to display high dimensional data in 2D.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/inv_cov_merged.png"/>
</p>

The figure above shows the resulting network of stocks constructed by thresholding the inverse covariance matrix on the left and the constructed precision matrix on the right. We can see three different clusters of corporate stocks, bonds and commodities linked according to the edges revealed by graph lasso algorithm.

**Gaussian Process Regression**

Gaussian Proccess (GP) is a way to perform Bayesian inference over functions. A GP assumes that p(f(x1),...,f(xn)) is jointly Gaussian with mean mu(x) and covariance Sigma(x) given by Sigma_ij = k(xi,xj), where k is a positive definite kernel.

<p align="center">
<img src="https://github.com/vsmolyakov/fin/blob/master/figures/market_gp.png"/>
</p>

The figure above shows GP regression applied to SP500 time series for a period of one year. Notice that the lack of future observations results in a constant prediction, i.e. the mean and the variance of the GP regressor do not change.


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
Quantopian