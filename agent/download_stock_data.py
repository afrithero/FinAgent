from stock.stock_loader import StockLoader

if __name__ == "__main__":
    # loader = StockLoader(stocks=["AAPL"], # list of stock ID
    #                      market="us", # currently only support us and tw stock market
    #                      start_date=(2024, 1), 
    #                      save_dir="../data/us_stock/")
    
    loader = StockLoader(["2330"], 
                         "tw", 
                         (2023, 1), 
                         "../data/tw_stock/")
    
    loader.run()