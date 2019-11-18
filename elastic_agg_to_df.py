# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:47:54 2019

@author: Tsabar
"""

import pandas as pd
import json
from pandas.io.json import json_normalize
import itertools
x=json.load(open(r'C:\Users\Tsabar\Documents\projects\anomaly sapir\try\try_elastic.json'))

X=json_normalize(x)

X=json_normalize(X['aggregations.2.buckets'])
X.columns

time_key = '2'
hosts_key = '3'

def process_row(time, row):
    host = row['key']
    row_data = { 'time': time, 'host': host }
    for key in row:
        if key == 'key' or key == 'doc_count': continue
        for value in row[key]:
            if isinstance(row[key][value], dict): continue
            row_data[key + '_' + value] = row[key][value]
    return row_data

def process_time_bucket(time_bucket):
    time = time_bucket['key']
    bucket_data = time_bucket[hosts_key]['buckets']
    rows = [process_row(time, x) for x in bucket_data]
    return rows
    

time_buckets = x['aggregations'][time_key]['buckets']
rows = [process_time_bucket(x) for x in time_buckets]
flat_rows = [y for x in rows for y in x]

df= pd.DataFrame(flat_rows)
time_temp=pd.to_datetime(df['time']*1000000)
