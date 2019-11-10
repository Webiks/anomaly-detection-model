from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import argparse
import json


def get_hit_dict(hit):
    hit_dict = hit.to_dict()
    hit_dict.update(hit.meta.to_dict())
    return hit_dict


def download_json(path, index, last_minutes, host, port, user, password):
    hosts = [{"host": host, "port": port}]
    http_auth = (user, password)
    elastic_client = Elasticsearch(hosts=hosts, http_auth=http_auth)
    search=Search(using=elastic_client)
    time_query = {
        "@timestamp": {
            "format": "strict_date_optional_time",
            "gte": "2019-11-09T08:00:00.000Z",
            "lte": "2019-11-09T10:00:00.000Z"
#            "gte": "now-{0}m".format(last_minutes),
#            "lt": "now"
        }
    }
    search = search.query('range', **time_query)
    total = search.count()
    print(total)
    results = [get_hit_dict(hit) for hit in search.scan()]

    print('downloaded')
    with open(path, 'w') as f:
        json.dump(results, f)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download elastic search last X minutes on an index')
    parser.add_argument('--path', '-o', help='JSON output path', type=str, default="C:/Users/Tsabar/projects/monitor/json/aws3.json", required=False)
    parser.add_argument('--index', '-i', help="ElasticSearch index", type=str, default='metricbeat-*', required=False)
    parser.add_argument('--lastMinutes', '-m', help="Last minutes to get data for", dest='last_minutes', type=int, default=0, required=False)
    parser.add_argument('--host', '-a', help="ElasticSearch Host's ip/address", type=str, default='elastic.monitor.net', required=False)
    parser.add_argument('--port', '-p', help="ElasticSearch Host's port", type=int, default=9200, required=False)
    parser.add_argument('--user', '-u', help="ElasticSearch username", type=str, default='elastic', required=False)
    parser.add_argument('--password', '-pw', help="ElasticSearch username's password", type=str, default='changeme', required=False)

    args = parser.parse_args()
    print(args)
    download_json(**vars(args))
