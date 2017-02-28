"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # Read in the orders file and get the symbols, start, and end date from the file.
    order_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True)
    symbols = order_df.Symbol.unique().tolist()
    start_date = order_df.index.min()
    end_date = order_df.index.max()

    # Read in prices for each of the symbols
    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Cash'] = pd.Series(np.ones(len(prices.index)), index=prices.index)

    # Initialize trades dataframe
    trades = prices.copy()
    trades.ix[:,:] = 0
    for index, row in order_df.iterrows():
        order_type = -1 if row['Order'] == 'SELL' else 1
        trades.loc[index, row['Symbol']] = trades.loc[index, row['Symbol']] + order_type * row['Shares']
        trades.loc[index, 'Cash'] = trades.loc[index, 'Cash'] + prices.loc[index, row['Symbol']] * row['Shares'] * -order_type

    # Initialize holdings dataframe
    holdings = prices.copy()
    holdings.ix[:,:] = 0
    holdings.ix[0, 'Cash'] = start_val

    # Calculate holdings
    holdings.ix[0,:] = holdings.ix[0,:] + trades.ix[0,:]
    for index, row in trades.ix[1:,:].iterrows():
        holdings.loc[index,:] = holdings.shift(1).loc[index,:] + row

    # Initialize values dataframe
    values = prices * holdings
    portvals = values.sum(axis=1)

    return portvals

def compute_portfolio_stats(daily_port_value):
    daily_returns = daily_port_value.copy()
    daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1
    daily_returns.ix[0,:] = 0
    daily_returns = daily_returns.ix[1:]
    adr = daily_returns.mean().iloc[0]
    sddr = daily_returns.std().iloc[0]
    sr = (adr/sddr) * math.sqrt(252)

    cr = (daily_port_value.iloc[-1,0] / daily_port_value.iloc[0,0]) - 1

    return cr, adr, sddr, sr

def get_dates(orders_file = "./orders/orders.csv"):
    order_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True)
    start_date = order_df.index.min()
    end_date = order_df.index.max()
    return start_date, end_date

def test_code():
    # this is a helper function you can use to test your code3
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-test.csv"
    sv = 1

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date, end_date = get_dates(of)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(pd.DataFrame(portvals))
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
