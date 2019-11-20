from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import argparse
from elastic_agg_to_df import build_generic_aggregations_dataframe
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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
            "gte": "now-{0}m".format(last_minutes),
            "lt": "now"
        }
    }
    search = search.query('range', **time_query)
    aggs = search.aggs.bucket('timestamp', 'date_histogram', field="@timestamp", interval="1m")
    aggs = aggs.bucket('host', 'terms', field="host.name.keyword", size=1000000000)
    aggs = aggs.bucket('city', 'terms', field="city.keyword", size=1000000000)
    aggs.metric('idle_sum', 'sum', field="system.cpu.idle.pct")
    aggs.metric('cpu_stats', 'extended_stats', field="system.cpu.system.pct")
    aggs.metric('cpu_steal', 'percentiles', field="system.cpu.steal.pct", percents=[25, 75], keyed=False)
    response = search.execute()
    df = build_generic_aggregations_dataframe(response)
    df.to_json(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download elastic search last X minutes on an index')
    parser.add_argument('--path', '-o', help='JSON output path', type=str, default="data.json", required=False)
    parser.add_argument('--index', '-i', help="ElasticSearch index", type=str, default='metricbeat-*', required=False)
    parser.add_argument('--lastMinutes', '-m', help="Last minutes to get data for", dest='last_minutes', type=int, default=5, required=False)
    parser.add_argument('--host', '-a', help="ElasticSearch Host's ip/address", type=str, default='elastic.monitor.net', required=False)
    parser.add_argument('--port', '-p', help="ElasticSearch Host's port", type=int, default=9200, required=False)
    parser.add_argument('--user', '-u', help="ElasticSearch username", type=str, default='elastic', required=False)
    parser.add_argument('--password', '-pw', help="ElasticSearch username's password", type=str, default='changeme', required=False)

    args = parser.parse_args()
    print(args)
    download_json(**vars(args))
