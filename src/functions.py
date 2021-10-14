import pandas as pd
import praw
from praw.models import MoreComments
from nltk import FreqDist
import nltk
from collections import defaultdict
from collections import Counter
#nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA 
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nltk.download('vader_lexicon') 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def create_stock_list(subreddit):
    """
    Get all comments from posts in the specified subreddit 
    containing the specified stock ticker

    Parameters
    -------------
    subreddit : str
 
    Returns
    -------------
    1. (defaultdict(list)) Stock ticker and all submission id's relating to that stock
    2. (list) stocks mentioned in top 100 posts 
    
    Example
    -------------
    get_comments(subid_of_picked_stock, reddit)
    """
    stocks = defaultdict(list)
    stocks_list = []
    
    for submission in subreddit.hot(limit=100):
        
        words = submission.title.split()
        ticker = list(set(filter(lambda word: word.lower().startswith('$') and word[1:].isalpha(), words)))
        
        if len(ticker) > 0:
            stocks[ticker[0]].append(submission.id)
            stocks_list.append(ticker[0])
            
        if len(ticker) > 1:
            stocks[ticker[1]].append(submission.id)
            stocks_list.append(ticker[1])

    return stocks, stocks_list



def remove_des(name):
    """
    remove 'Class A Common Stock' and 'Common Stock' from listing names
    """
    if name.endswith("Class A Common Stock"):
        name = name[:-21]
    if name.endswith("Common Stock"):
        name = name[:-13]
           
    return name


def get_comments(tickers, reddit):
    """
    Get all comments from posts in the specified subreddit 
    containing the specified stock ticker

    Parameters
    -------------
    tickers : list
    
    reddit : reddit API 
 
    Returns
    -------------
    (list) All comments for posts relating to the specified ticker
    
    Example
    -------------
    get_comments(subid_of_picked_stock, reddit)
    """
    comments_all_posts = []
    for i in tickers:
        post = reddit.submission(id=i)
        post.comments.replace_more(limit=None)
        for comments in post.comments.list():
            comments_all_posts.append(comments.body)
    return comments_all_posts


def preprocess(
    text,
    min_token_len=2,
    irrelevant_pos=["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE"],
):
    """
    Given text, min_token_len, and irrelevant_pos carry out preprocessing of the text
    and return a preprocessed string.

    Parameters
    -------------
    text : (str)
        text to be preprocessed
    min_token_len : (int)
        minimum token length required to keep a token in the preprocessed text
    irrelevant_pos : (list)
        a list of irrelevant pos tags

    Returns
    -------------
    (str) the preprocessed text
    """
    clean_text = []

    for token in text:
        if (
            token.is_stop == False                # Check if it's not a stopword
            and len(token) > min_token_len        # Check if the word meets minimum threshold
            and token.pos_ not in irrelevant_pos  # Check if the POS is in the acceptable POS tags
            and not token.like_email              # Check if it's not an email
            and not token.like_url                # Check if it's not a url
            and token.is_alpha                    # Check if consist of alphabetic characters
            
        
        ):           
            lemma = token.lemma_                  # Take the lemma of the word
            clean_text.append(lemma.lower())
    return " ".join(clean_text)


def vader_sentiment(cleaned_text:list):
    """
    Sentiment analysis using Vader
    """
    sia = SIA()
    results = []
    for sentences in cleaned_text:
        pol_score = sia.polarity_scores(sentences)
        pol_score["words"] = sentences
        results.append(pol_score)
        
    pd.set_option('display.max_columns', None, 'max_colwidth', None)
    df = pd.DataFrame.from_records(results)
    
    #set threshold for sentiment (pos, neg, neu)
    df["sentiment"] = "neutral"
    df.loc[df['compound'] > 0.10, 'sentiment'] = "positive"
    df.loc[df['compound'] < -0.10, 'sentiment'] = "negative"

    return df



def sentiment_barchart(df): 
    """
    Given a dataframe with the sentences/words and sentiment, returns a barchart
    showing distribution of sentiments (pos, neu, neg)

    Parameters
    -------------
    df : pd.DataFrame
    
    Returns
    -------------
    Plotly Bar Chart
    
    Example
    -------------
    sentiment_barchart(sentiment_df)
    """
    layout = go.Layout(
        margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0,  #top margin
        pad=0
        )
    )
    
    fig = px.bar(df['sentiment'].value_counts(), y='sentiment', x=['neutral', 'negative', 'positive'],
                color=["grey", "red", "green"], 
                color_discrete_map="identity",
                labels={"x": "Sentiment",
                        "sentiment": "count of records"
                        }
                )
    fig.update_layout(showlegend=False, autosize=True)
    
    return fig


def pie_chart(df):
    """
    Given a dataframe stock tickers and counts, return pie chart for distribution

    Parameters
    -------------
    df : pd.DataFrame
    
    Returns
    -------------
    Plotly Pie Chart
    
    Example
    -------------
    pie_chart(stock_df)
    """
    
    layout = go.Layout(
        margin=go.layout.Margin(
        l=50, #left margin
        r=50, #right margin
        b=50, #bottom margin
        t=50,  #top margin
        pad=0
        )
    )

    labels = df[:10]["Symbol"]
    values = df[:10]["Number of posts"]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)], layout = layout)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, autosize=True)
    
    return fig


def preprocess_word(
    text,
    min_token_len=2,
    irrelevant_pos=["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE"],
):
    """
    Given text, min_token_len, and irrelevant_pos carry out preprocessing of the text
    and return a preprocessed string.

    Parameters
    -------------
    text : (str)
        text to be preprocessed
    min_token_len : (int)
        minimum token length required to keep a token in the preprocessed text
    irrelevant_pos : (list)
        a list of irrelevant pos tags

    Returns
    -------------
    (str) the preprocessed text
    """
    clean_text = []

    for token in text:
        if (
            token.is_stop == False                # Check if it's not a stopword
            and len(token) > min_token_len        # Check if the word meets minimum threshold
            and token.pos_ not in irrelevant_pos  # Check if the POS is in the acceptable POS tags
            and not token.like_email              # Check if it's not an email
            and not token.like_url                # Check if it's not a url
            and token.is_alpha                    # Check if consist of alphabetic characters
            
        
        ):           
            lemma = token.lemma_                  # Take the lemma of the word
            clean_text.append(lemma.lower())
    return clean_text



def word_cloud(df, sentiment):
    """
    Given a dataframe with the word and sentiment, return a word cloud

    Parameters
    -------------
    df : pd.DataFrame
    
    sentiment : str
        "positive", "neutral", or "negative"
 
    Returns
    -------------
    word cloud
    
    Example
    -------------
    word_cloud(df, 'positive')
    """
    words = df.loc[df['sentiment'] == sentiment]['words']
    freq = FreqDist(words).most_common(20)
    word_freqs = ' , '.join([str(p) for p in freq])
    
    color = "Greens"
    if sentiment == "negative":
        color = "Reds"
    
    wordcloud = WordCloud(background_color = 'white', width=800, height=400, colormap = color).generate(word_freqs)
    plt.imshow(wordcloud, interpolation ='bilinear', aspect="auto")
     
    return wordcloud