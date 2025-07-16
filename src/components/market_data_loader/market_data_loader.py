import yfinance as yf
import pandas as pd

class MarketDataLoader:
    def __init__(self, tickers: list = ['^GSPC', '^VIX'], start: str = "2015-01-01"):
        self.tickers = tickers
        self.start = start

    def load(self) -> pd.DataFrame:

        data = yf.download(self.tickers, start=self.start)['Close']
        data = data.dropna()

        # Calculate daily returns
        data['SP500_ret'] = data['^GSPC'].pct_change()
        data['VIX_change'] = data['^VIX'].pct_change()
        data = data.dropna()

        return data