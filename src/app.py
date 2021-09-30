import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import praw
import time
import functions as fn
import spacy
from decouple import Config, RepositoryEnv
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
st.set_page_config(layout="wide")


"""
# r/Wallstreetbets Sentiment Analysis
Created by: Jordan L.
"""

"""
### Companies that are currently *Hot* on WSB :fire: :
"""

from decouple import Config, RepositoryEnv

DOTENV_FILE = 'reddit.env'
env_config = Config(RepositoryEnv(DOTENV_FILE))
SECRET_USER = env_config.get('client_id')
SECRET_KEY = env_config.get('client_secret')

#connect to reddit 
reddit = praw.Reddit(client_id = SECRET_USER,
                    client_secret= SECRET_KEY,
                    user_agent = "ua")
#select subrreddit
subreddit = reddit.subreddit('wallstreetbets')

#call function to create dictionary(stocks and ID) and list(count of mentions)
stocks_and_subid, stocks_list = fn.create_stock_list(subreddit)

#read in stock listings
nyse_nasdaq_listings = pd.read_csv("~/wallstreetbets_sentiment_analysis/data/nyse_nasdaq_listings.csv")

#----------------DataFrame for Hot Stocks--------------

top_mentions = (pd.DataFrame.from_dict(Counter(stocks_list), orient='index').
                reset_index().
                rename(columns={0: "Number of posts", "index": "Symbol"}).
                sort_values("Number of posts", ascending = False))

top_mentions_merged = top_mentions.merge(nyse_nasdaq_listings, on = 'Symbol', how = 'left')


top_mentions_merged

#--------------------Sidebar-------------------------------

picked_stock = st.sidebar.selectbox(
    'Perform sentiment analysis on:',
     top_mentions_merged['Symbol'])

#---------------Process Comments and get sentiment---------------
with (st.spinner("Please wait, scraping data and analyzing " 
    + str(int((top_mentions_merged[top_mentions_merged['Symbol'] == picked_stock]["Number of posts"]))) 
    + ' posts of '+ str(picked_stock))):

    subid_of_picked_stock = stocks_and_subid[picked_stock]
    #call function to get comments from posts of selected stock
    comments_all_posts = fn.get_comments(subid_of_picked_stock, reddit)
    #call function to preprocess text
    cleaned_text_all_posts = [fn.preprocess(text) for text in nlp.pipe(comments_all_posts)]
    # call function to get sentiments with Vader
    df_all_posts = fn.vader_sentiment(cleaned_text_all_posts)

(st.success('Finished performing analysis of comments on ' 
    + str(int((top_mentions_merged[top_mentions_merged['Symbol'] == picked_stock]["Number of posts"]))) 
    + ' posts of ' + str(picked_stock)))
st.balloons()

#---------First layer of plots (bar chart and pie chart)------------
bar_chart = fn.sentiment_barchart(df_all_posts)
pie_chart = fn.pie_chart(top_mentions_merged)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribution of Top 10 mentions")
    st.plotly_chart(pie_chart, use_container_width = True)
with col2:
    st.subheader("Sentiment of comments for: " + str(picked_stock))
    st.plotly_chart(bar_chart, use_container_width = True)

#---------- Second layer of plots (wordclouds) ---------------------
positive_wc = fn.word_cloud(df_all_posts, "positive")
negative_wc = fn.word_cloud(df_all_posts, "negative")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Positive Sentiment Word Cloud")
    st.image(positive_wc.to_array())
with col2:
    st.subheader("Negative Sentiment Word Cloud")
    st.image(negative_wc.to_array())


