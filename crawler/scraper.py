import twint
from utils import get_start_date


def scrape_tweets():
    """
    function to scrape tweets and put them inside a pandas dataframe
    """
    c = twint.Config()
    c.Search = "covid vaccin lang:fr"
    c.Since = get_start_date()
    c.Pandas = True

    twint.run.Search(c)
    tweets = twint.storage.panda.Tweets_df

    return tweets
