import re
import requests
from copy import deepcopy
from datetime import datetime, timedelta
from string import punctuation
from tqdm import tqdm
import spacy
import fr_core_news_sm
from simpletransformers.classification import ClassificationModel
from elasticsearch import Elasticsearch


# Loading spacy French language model


def load_spacy():
    nlp = fr_core_news_sm.load(disable=["parser", "tagger"])
    suffixes = nlp.Defaults.suffixes + [r"\d*?[\.,]?\d*\%"]
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    return nlp


nlp = load_spacy()
print("spacy model loaded with french NER")


# Loading the sentiment analysis model

model = ClassificationModel("bert", "/model", use_cuda=False)


# Loading french stopwords

extra_stopwords = ["qu'", "qu", "«", "»", "...", "'", ".."]


def download_stopwords(url):
    response = requests.get(url)
    stopwords = response.content.decode("utf-8").split("\n")
    return stopwords


def get_stopwords(urls):
    STOPWORDS = []
    for url in urls:
        STOPWORDS += download_stopwords(url)
    STOPWORDS += extra_stopwords
    return STOPWORDS


stopwords_urls = [
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt"
]

STOPWORDS = get_stopwords(stopwords_urls)
print("French stopwords loaded ...")

# Setting up ES

es = Elasticsearch("elasticsearch")


# return max date from ES


def get_start_date():
    r = es.search(
        index="tweets",
        params={"size": 1},
        body={"aggs": {"max_created_date": {"max": {"field": "created_date"}}}},
    )
    d = r["hits"]["hits"][0]["_source"]["created_at"]
    d = datetime.strptime(d, "%Y-%m-%dT%H:%M:%S%z")
    d = d.replace(tzinfo=None)
    d = d + timedelta(hours=1)
    d = str(d)
    return d


# Tokenization and tweet cleaning


def strip_url(text):
    return re.sub(r"http\S+", "", text)


def strip_mentions(text):
    text = re.sub(r"@\w+", "", text)
    return text


def clean_tweets(tweet):
    tweet = strip_mentions(tweet)
    tweet = strip_url(tweet)
    tweet = tweet.replace("’", "'")
    tweet = re.sub(r"covid[\w-]*19", "covid-19", tweet, flags=re.IGNORECASE)
    return tweet


def tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def clean_tokens(tokens):
    tokens = [token.lower().strip() for token in tokens]
    tokens = list(filter(lambda token: token not in punctuation, tokens))
    tokens = list(filter(lambda token: token not in STOPWORDS, tokens))
    tokens = [
        token for token in tokens if not ((len(token) == 2) & (token.endswith("'")))
    ]
    tokens = [
        token for token in tokens if not ((len(token) == 2) & (token.endswith("’")))
    ]
    tokens = [token for token in tokens if token != ""]
    tokens = [token for token in tokens if token not in ["..", "'", '"']]
    return tokens


# NER


def extract_entities(tweet):
    doc = nlp(tweet)
    entities = doc.ents

    output = {"LOC": [], "PER": [], "ORG": [], "MISC": []}

    for entity in entities:
        entity_label = entity.label_
        output[entity_label].append(entity.text)
    return output


def clean_entities(entities):
    for i, entity_dict in enumerate(entities):

        locations = deepcopy(entity_dict["LOC"])
        orgas = deepcopy(entity_dict["ORG"])
        persons = deepcopy(entity_dict["PER"])
        miscs = deepcopy(entity_dict["MISC"])

        for location in locations:
            if "covid" in location.lower():
                entities[i]["LOC"].remove(location)
                entities[i]["MISC"].append(location)

            elif location.lower() in [
                "vaccin",
                "💉",
                "ebola",
                "grippe",
                "la grippe",
                "sida",
            ]:
                entities[i]["LOC"].remove(location)
                entities[i]["MISC"].append(location)

            elif location.lower() in [
                "moderna",
                "sanofi",
                "sanofi france",
                "pfizer",
                "sinopharm",
                "laboratoire Moderna",
                "valneva",
            ]:
                entities[i]["LOC"].remove(location)
                entities[i]["ORG"].append(location)

            elif location.lower() in ["trump", "biden"]:
                entities[i]["LOC"].remove(location)
                entities[i]["PER"].append(location)

            elif len(location) == 1:
                entities[i]["LOC"].remove(location)

        for orga in orgas:
            if "covid" in orga.lower():
                entities[i]["ORG"].remove(orga)
                entities[i]["MISC"].append(orga)

            elif orga.lower() in [
                "pcr",
                "grippe",
            ]:
                entities[i]["ORG"].remove(orga)
                entities[i]["MISC"].append(orga)

            elif orga.lower() in [
                "new post",
                "plus",
                "by by",
                "siècle",
                "➡",
                "oui",
                "preuve",
                "soutenez",
                "y’",
                "bravo",
                "alerte",
                "nuls",
                "flash",
            ]:
                entities[i]["ORG"].remove(orga)

            elif orga.lower() in ["biden", "trump", "macron", "elon musk"]:
                entities[i]["ORG"].remove(orga)
                entities[i]["PER"].append(orga)

            elif len(orga) == 1:
                entities[i]["ORG"].remove(orga)

        for per in persons:
            if "covid" in per.lower():
                entities[i]["PER"].remove(per)
                entities[i]["MISC"].append(per)

            elif per.lower() in [
                "biontech",
                "moderna",
                "pfizer",
                "église shincheonji",
                "medicago",
                "astra zeneca",
                "l’inserm",
                "mali web",
                "reuters",
            ]:
                entities[i]["PER"].remove(per)
                entities[i]["ORG"].append(per)

            elif per.lower() in [
                "j’",
                "bonjour",
                "regardez",
                "merci",
                "avez",
                "faudra",
                "vidéo",
                "| jdm",
                "mdr",
                "retrouvez",
                "ptdr",
                "iii",
                "bonne",
                "normal",
                "hâte",
                "|\xa0",
                "ma mère",
                "hier",
                "lome infos",
                "jr",
                "réveillez",
                "hold_up",
                "vaccinez",
                "vacciner",
                "meme",
                "😂😂😂",
                "🤔🤔",
            ]:
                entities[i]["PER"].remove(per)

            elif per.lower() in [
                "vaccin",
                "grippe",
                "coronavirus",
                "virus",
                "coronavirus\xa0",
                "arnm",
            ]:
                entities[i]["PER"].remove(per)
                entities[i]["MISC"].append(per)

            elif per.lower() in ["🇬🇧"]:
                entities[i]["PER"].remove(per)
                entities[i]["LOC"].append(per)

            elif len(per) == 1:
                entities[i]["PER"].remove(per)

        for misc in miscs:
            if "covid" in misc.lower():
                new_misc = re.sub(r"covid[\w*\s]19", "covid-19", misc.lower())
                entities[i]["MISC"].remove(misc)
                entities[i]["MISC"].append(new_misc)

            elif misc.lower() in ["pfizer", "moderna", "sanofi", "laboratoire pfizer"]:
                entities[i]["MISC"].remove(misc)
                entities[i]["ORG"].append(misc)

            elif misc.lower() in ["trump", "biden"]:
                entities[i]["MISC"].remove(misc)
                entities[i]["PER"].append(misc)

            elif misc.lower() in ["c’", "Ça", "qu’", "s’", "y’"]:
                entities[i]["MISC"].remove(misc)

            elif len(misc) in [1, 2]:
                entities[i]["MISC"].remove(misc)

    return entities


# Sentiment Analysis


def compute_sentiment(tweets):
    predictions = model.predict(tweets)
    sentiments = predictions[0]
    confidences = predictions[1]
    return sentiments, confidences


# Dumping data to ES

columns = [
    "id",
    "created_at",
    "username",
    "tweet",
    "replies_count",
    "retweets_count",
    "likes_count",
    "hashtags",
    "clean_tokens",
    "LOC",
    "PER",
    "ORG",
    "MISC",
    "sentiment",
    "confidence",
]


def push_data_to_es(tweets):
    tweets = tweets[columns]
    int_columns = [c for c in tweets.columns if str(tweets[c].dtype) in ["int64"]]
    float_columns = [c for c in tweets.columns if str(tweets[c].dtype) in ["float64"]]

    for i in tqdm(range(len(tweets))):
        row = tweets.loc[i].to_dict()
        for k in row:
            if k in int_columns:
                row[k] = int(row[k])
            elif k in float_columns:
                row[k] = float(row[k])

        row["created_at"] = row["created_at"].strftime("%Y-%m-%dT%H:%M:%S%z")

        es.index(index="tweets", doc_type="tweet", id=row["id"], body=row)
