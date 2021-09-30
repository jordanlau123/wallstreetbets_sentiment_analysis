import pandas as pd
import numpy as np
import functions as fn


# readn in raw data from nasdaq website: https://www.nasdaq.com/market-activity/stocks/screener
nyse = pd.read_csv("~/wallstreetbets_sentiment_analysis/data/nyse_stocks.csv")
nasdaq = pd.read_csv("~/wallstreetbets_sentiment_analysis/data/nasdaq_stocks.csv")

# combine and preprocess data
nyse_nasdaq_listings = pd.concat([nasdaq, nyse])
nyse_nasdaq_listings['Symbol'] = '$' + nyse_nasdaq_listings['Symbol'].astype(str)
nyse_nasdaq_listings = nyse_nasdaq_listings[["Symbol", "Name", "Country", "Sector", "Industry"]]
nyse_nasdaq_listings["Name"] = nyse_nasdaq_listings["Name"].apply(fn.remove_des)


# convert to csv 
nyse_nasdaq_listings.to_csv("~/wallstreetbets_sentiment_analysis/data/nyse_nasdaq_listings.csv", index= False)







