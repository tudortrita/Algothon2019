import quandl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


def getTickers(ticker):
    productCodes = STOCK_MPC[STOCK_MPC.ticker == ticker].product_code.drop_duplicates()
    supplyChain = []
    tickers = []
    for productCode in productCodes:
        supplyChain += list(filteredSCREL[filteredSCREL['primary_code'] == productCode].related_code.drop_duplicates())
        supplyChain = list(set(supplyChain))

    for product in supplyChain:
        tickers += list(STOCK_MPC[STOCK_MPC.product_code == product].ticker.drop_duplicates())

        return list(set(tickers))


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        # shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        df = pd.concat([datax, shiftedy], axis=1)
        return np.array(df.corr())[0, 1]
    else:
        df = pd.concat([datax, datay], axis=1)
        return np.array(df.corr())[0, 1]


MPC = pd.read_csv("CSO_MPC.csv")
DYCoverage = pd.read_csv("DY_SPA.csv")

wendi_api = ""
quandl.ApiConfig.api_key = wendi_api

tudor_api = ""
quandl.ApiConfig.api_key = tudor_api

MPC = MPC[['sec_code', "product_code"]]
STOCK = quandl.get_table('CSO/STOCK', qopts={"columns":["sec_code", "ticker"]}, paginate=True)
SCREL =  quandl.get_table('CSO/SCREL', qopts={"columns":["primary_code", "related_code", "relationship", ]}, paginate=True)

filteredSCREL = SCREL[SCREL.relationship == -1]
STOCK_MPC = pd.merge(STOCK, MPC)

# STOCK_TICKERS = STOCK.ticker.drop_duplicates()
# startTickersList = list(set(DYCoverage.ticker).intersection(set(STOCK_TICKERS)))
# print(startTickersList)

stock = "600525"
# Begin with stock 600525:
tickers = getTickers(stock)
availableTickers = list(set(DYCoverage.ticker).intersection(set(tickers)))

# Compute cross-correlation of each:
timeseries1 = quandl.get_table('DY/SPA', ticker=stock, qopts={"columns":["date", "close"]}, paginate=True)
timeseries1.set_index('date', inplace=True)
timeseries1 = timeseries1.astype(float)
testValueTimeseries = timeseries1.iloc[-1]
timeseries1 = timeseries1.iloc[:-1]

# mappings = {"600525":availableTickers}
# timeseries1.plot()

print(len(availableTickers))
weightedSum = 0

for t in availableTickers:
    tseries = quandl.get_table('DY/SPA', ticker=t, qopts={"columns":["date", "close"]}, paginate=True)
    tseries.set_index('date', inplace=True)
    tseries = tseries.astype(float)
    tseries1 = tseries.iloc[:-1]
    testTseries = tseries.iloc[-1]

    # Accounting for missing dates, etc:
    timeseries1 = timeseries1.reindex(tseries1.index)


    t_steps = 30
    rs = [crosscorr(timeseries1, tseries1, lag=lag, wrap=True) for lag in range(-t_steps, 0)]


    offset = np.ceil(len(rs)/2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(range(-t_steps, 0), rs)
    ax.axvline(np.ceil(len(rs)/2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', xlim=[-t_steps, -1], xlabel='Offset', ylabel='Pearson r')
    plt.legend()
    plt.show()

    max_y = max(rs)  # Find the maximum y value
    max_x = range(-t_steps, 0)[rs.index(max_y)]  # Find the x value corresponding to the maximum y value
    print("lag, correlation: ", max_x, max_y)
    if not np.isnan(max_y):
        weightedSum += max_y*tseries.close[max_x]

print(weightedSum/len(availableTickers), testValueTimeseries)






# orgs = quandl.get_table('CSO/STOCK', qopts={"columns":["org", "ticker"]}, paginate=True)
# for ticker in tickers:
#     print(orgs[orgs.ticker == ticker].org)



# stockPrices = quandl.get_table('DY/SPA', ticker = availableTickers[0], qopts={"columns":["date", "close"]}, paginate=True)



# close = stockPrices['close'].astype(float)
# log_rets = np.log(close) - np.log(close.shift(1))
# volatility = log_rets.shift(1).ewm(span=252).std()
# rets = log_rets / volatility
# slow_period = 300
# fast_period = 150
# slow_rets = rets.ewm(span = slow_period).mean()
# fast_rets = rets.ewm(span = fast_period).mean()
# diff = fast_rets - slow_rets
# raw_macd = diff
# exp_diff = np.sqrt(np.square(diff).ewm(span=60).mean())
# rms_macd = diff / exp_diff
# std_diff = diff.ewm(span=63).std()
# std_macd = diff / std_diff
