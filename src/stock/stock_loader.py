import concurrent.futures
import os
import pandas as pd
from twstock import Stock
import yfinance as yf

class StockLoader:
    def __init__(self, stocks, market, start_date, save_dir):
        self.stocks = stocks # e.g. ["2330"]
        self.market = market # "us" or "tw"
        self.start_date = start_date
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_one_stock_to_csv(self, stock_id):
        print(f"Processing on: {stock_id}")
        try:
            if self.market == "us":
                start = f"{self.start_date[0]:04d}-{self.start_date[1]:02d}-01"
                df = yf.download(stock_id, start=start)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)                
                
                df = df.reset_index()
            
            elif self.market == "tw":
                stock = Stock(stock_id)
                stock.fetch_from(year=self.start_date[0], month=self.start_date[1])
                data_dicts = [d._asdict() for d in stock.data]
                df = pd.DataFrame(data_dicts)
                df = df[["date", "open", "high", "low", "close", "capacity"]]
                df = df.rename({"capacity": "Volume",
                                "date": "Date",
                                "close": "Close",
                                "open": "Open",
                                "high": "High",
                                "low": "Low"}, axis=1)
            else:
                raise ValueError("Currently only supports 'tw' or 'us' market.")
            
            filepath = os.path.join(self.save_dir, f"{stock_id}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved on: {filepath}")
        
        except Exception as e:
            print(f"Error processing {stock_id}: {e}")
    
    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.save_one_stock_to_csv, stock_id)
                for stock_id in self.stocks
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error fetching data for {e}")

        print("Finished all runs.")