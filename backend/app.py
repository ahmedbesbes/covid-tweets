import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from tqdm import tqdm

if __name__ == "__main__":
    print("loading the raw data ")
    with open("/data/es_data.json") as f:
        data = json.load(f)

    print("creating Elasticsearch index")
    es = Elasticsearch("elasticsearch")

    try:
        es.indices.delete(index="tweets")
        print("deleted tweets index")
    except:
        print("tweets index already deleted")

    es.indices.create(index="tweets")

    def generate_actions():
        for doc in tqdm(data):
            yield ({
                "_index": "tweets",
                "_source": doc,
                "_type": "tweet",
                "_id": doc["id"]
            })

    for success, info in parallel_bulk(es, generate_actions()):
        if not success:
            print(f"A document failed: {info}")

    print("data dumped successfully in Elasticsearch")
