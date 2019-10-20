# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:42:25 2019

@author: Desai
"""

import quandl
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


quandl.ApiConfig.api_key = ""

data = quandl.get_table('GSCR/GSPLY',
                        publish_date='2018-05-18',
                        supplier_exchange='TSE')

ref = quandl.get_table('GSCR/GSREF',
                        publish_date='2018-05-18')
#
#f = pd.merge(pd.DataFrame(data['customer_isin']), ref, how="left", left_on = "customer_isin",
#                                right_on = "isin")

china_stocks = quandl.get_table('DY/SPA'
                                , date='2017-12-04')

china_names = quandl.get_table('CSO/STOCK')


quandl.ApiConfig.api_key = ""
china_ggdp = quandl.get('SGE/CHNGGR', start_date='2005-12-31',
                       end_date='2020-02-29')
#
#model = sm.tsa.MarkovRegression(china_ggdp, k_regimes=2)
#res = model.fit()


US = pd.read_csv("data_USA.csv").drop('SGE/USACAGDP', axis = 1)
#x = US.columns
#GDP = US[[x[0],[s for s in x if 'GDP' in s][0]]].dropna()
#df = GDP.merge(US).drop([s for s in x if 'GDP' in s][0], axis=1)
#df = df.fillna(df.mean())

Tudor = pd.read_csv("GDP.csv").rename(columns = {'DATE':'Date'}).dropna()
df = Tudor.merge(US, how = "left", on="Date")


mod = sm.tsa.MarkovRegression(Tudor['GDP GROWTH'].dropna(), k_regimes = 2).fit()
print(mod.summary())
