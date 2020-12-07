# from tqdm import tqdm
# import pandas as pd
# from scraper import scrape_tweets
# from utils import (
#     clean_tweets,
#     tokenize,
#     clean_tokens,
#     extract_entities,
#     clean_entities,
#     compute_sentiment,
#     push_data_to_es,
# )

# from elasticsearch import Elasticsearch


# if __name__ == "__main__":
#     tweets = scrape_tweets()

#     tweets["tweet_clean"] = tweets["tweet"].map(clean_tweets)
#     print("tweets cleaned")

#     tweets["tokens"] = tweets["tweet_clean"].map(tokenize)
#     print("tweets tokenized")

#     tweets["clean_tokens"] = tweets["tokens"].map(clean_tokens)
#     print("tokens cleaned")

#     entities = []
#     for i in tqdm(range(len(tweets))):
#         ents = extract_entities(tweets["tweet_clean"][i])
#         entities.append(ents)
#     entities = clean_entities(entities)
#     tweets = pd.concat([tweets, pd.DataFrame(entities)], axis=1)
#     print("entities extracted")

#     sentiments, confidences = compute_sentiment(tweets["tweet_clean"].tolist())
#     tweets["sentiment"] = sentiments
#     tweets["confidence"] = confidences
#     print("sentiment score and confidence computed")

#     push_data_to_es(tweets)
#     print("data dumped to ES successfully!")

from elasticsearch import Elasticsearch

if __name__ == "__main__":
    es = Elasticsearch("elasticsearch")

    r = es.search(
        index="tweets",
        params={"size": 1},
        body={"aggs": {"max_created_date": {"max": {"field": "created_date"}}}},
    )

    print("output :", r)
