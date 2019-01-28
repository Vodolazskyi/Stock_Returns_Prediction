"""Metrics to assess performance of a trading strategy"""


import pandas as pd
import numpy as np

from scipy.stats import ttest_1samp
from sklearn.metrics import roc_auc_score


def max_drawdown(cum_returns, invert=True):
    """Function to calculate maximum drawdown.

    A maximum drawdown (MDD) is the maximum loss from a peak to a trough
    of a portfolio, before a new peak is attained.
    Maximum Drawdown is expressed in percentage terms and computed as:
    MDD = (Trough Value–Peak Value) / Peak Value

    Parameters
    ----------
    cum_returns : 1d array-like
        Cumulative returns of a portfolio.

    invert : bool, optional (defaul=True)
        If ``False``, return the negative number.
        Otherwise, return maximum drawdown as a positive number.
        
    Returns
    -------
    mdd : float
        Maximum drawdown
    """

    highest = [0]
    ret_idx = cum_returns.index
    drawdown = pd.Series(index=ret_idx)

    for t in range(1, len(ret_idx)):
        cur_highest = max(highest[t-1], cum_returns[t])
        highest.append(cur_highest)
        drawdown[t] = (1+cum_returns[t])/(1+highest[t]) - 1
        
    if invert:
        return -1 * drawdown.min()
    else:
        return drawdown.min()


def onesided_ttest(returns, mean=0, alternative='greater'):
    """Function to calculate p-value of a one-sided t-test

    Parameters
    ----------
    returns : 1d array-like
        Stock returns of a trading strategy.
    
    mean : float, optional (default=0)
        Mean value for testing.

    alternative : str, optional (default='greater'), {'greater', 'less'}
        Define the alternative hypothesis for t-test

    Returns
    -------
    pvalue : float
        one-tailed p-value
    """

    ttest = ttest_1samp(returns, mean)
    if alternative == 'greater':
        if ttest[0] > 0:
            return ttest[1]/2
        else:
            return 1 - ttest[1]/2
    
    if alternative == 'less':
        if ttest[0] > 0:
            return 1 - ttest[1]/2
        else:
            return ttest[1]/2


def Sharpe(returns, n=252):
    """Function to calculate Sharpe ratio

    The Sharpe ratio is the average return earned in excess of the
    risk-free rate per unit of volatility or total risk.

    Parameters
    ----------
    returns: 1-d array-like
        Returns of a trading strategy
    
    n : int, optional (default=252)
        Number of time periods to calculate Sharpe ratio on different
        time basis. For example, default n=252 calculatutes annualized
        Sharpe ratio for daily returns
    
    Returns
    -------
    sharpe : float
        Sharpe ratio
    """

    sharpe = returns.mean() * np.sqrt(n) / returns.std()
    return sharpe


def gini_coef(y_true, y_pred):
    """
    Function to calculate Gini coefficient
    """
    return 2*roc_auc_score(y_true, y_pred)-1