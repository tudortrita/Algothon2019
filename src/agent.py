""" Main trading agent.

We have daily time-series data
"""
import pyalgotrade
import quandl

class PnlSnapshot:
    def __init__(self, ticker, buy_or_sell, traded_price, traded_quantity):
        self.m_ticker = ticker
        self.m_net_position = 0
        self.m_avg_open_price = 0
        self.m_net_investment = 0
        self.m_realized_pnl = 0
        self.m_unrealized_pnl = 0
        self.m_total_pnl = 0
        self.update_by_tradefeed(buy_or_sell, traded_price, traded_quantity)

    # buy_or_sell: 1 is buy, 2 is sell
    def update_by_tradefeed(self, buy_or_sell, traded_price, traded_quantity):
        # buy: positive position, sell: negative position
        quantity_with_direction = traded_quantity if buy_or_sell == 1 else (-1) * traded_quantity
        is_still_open = (self.m_net_position * quantity_with_direction) >= 0
        # net investment
        self.m_net_investment = max( self.m_net_investment, abs( self.m_net_position * self.m_avg_open_price  ) )
        # realized pnl
        if not is_still_open:
            # Remember to keep the sign as the net position
            self.m_realized_pnl += ( traded_price - self.m_avg_open_price ) * min(abs(quantity_with_direction), abs(self.m_net_position)) * ( abs(self.m_net_position) / self.m_net_position )
        # total pnl
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl
        # avg open price
        if is_still_open:
            self.m_avg_open_price = ( ( self.m_avg_open_price * self.m_net_position ) + ( traded_price * quantity_with_direction ) ) / ( self.m_net_position + quantity_with_direction )
        else:
            # Check if it is close-and-open
            if traded_quantity > abs(self.m_net_position):
                # net position
                self.m_avg_open_price = traded_price
        self.m_net_position += quantity_with_direction

    def update_by_marketdata(self, last_price):
        self.m_unrealized_pnl = ( last_price - self.m_avg_open_price ) * self.m_net_position
        self.m_total_pnl = self.m_realized_pnl + self.m_unrealized_pnl

USD = PnlSnapshot("USDGBP", 2, 1000, 1)
USD.update_by_tradefeed(1, 1040, 1)


USD.update_by_tradefeed(1, 1010, 10)
USD.m_total_pnl


class Agent():

    def __init__(self, balance=0.0, total_pnl=0.0):
        self.balance = balance
        self.total_pnl = total_pnl


def main():
    quandl.ApiConfig.api_key = ""

    # Chinese stock data:
    df = quandl.get_table('DY/SPA', ticker='600170')
    df2 = quandl.get_table('CSO/SCREL', paginate=True)
    df3 = quandl.get_table('CSO/STOCK', paginate=True)

    pd.DataFrame(df2['primary_product'].unique()).to_csv("primary_products_unique.csv")
    df2.head()
    df3.to_csv("cso-stock.csv")
    df['adj_close'].astype(float).plot()
    df['name']

    # Ticker column corresponds to the column codings.
    df[['close'. 'turnover_value']].astype(float).plot(figsize=(13, 8))
    df[['date', 'close']]
    df.to_csv("test.csv")
    return

if __name__=='__main__':
    backtest = False

    main()
