# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:19:09 2019

@author: Tsabar
"""

from sklearn.ensemble import IsolationForest as IF
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json 
from pandas.io.json import json_normalize
import pandas as pd
import datetime
import bisect
import collections

### display set
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

## reading files - need to change to cloud and not locally

with open (r'C:\Users\Tsabar\projects\monitor\json\test11.json') as json_file: X=json.load(json_file)
with open (r'C:\Users\Tsabar\projects\monitor\json\test6.json') as json_file: X=json.load(json_file)
with open (r'C:\Users\Tsabar\projects\monitor\json\test10.json') as json_file: X2=json.load(json_file)
with open (r'C:\Users\Tsabar\projects\monitor\json\test6.json') as json_file: X6=json.load(json_file)
X=json_normalize(X)
X11=json_normalize(X11)
X2=json_normalize(X2)
X6=json_normalize(X6)
#X1=json_normalize(X)
#X3=json_normalize(X)
X['@timestamp'].tail(10)
X2['@timestamp'].head(10)
X0=pd.concat([X,X2])
d={}
df_list=[]
for i in range(0,3):
    df_list.append('X'+str(i))
#    for i in df_list:
#        d[i]=json_normalize(X)
    for c in df_list:
        exec('{} = json_normalize(X)'.format(c))

['X0'].head()
X0['meta.index'].head(2)

#### Pandas profiling on coolab

### investigate on my own

#X2=X2.sort_values(by=['_source.@timestamp'])
#temp.to_csv('temporary_new_monitor')
#sum(X2.groupby('_id')['_id'].nunique())
#X2.groupby('_source.process.name')['_source.process.name','_id','_source.system.process.cpu.start_time'].nunique()
#list(X6.columns)
#X2.groupby( '_source.user.name')[ '_source.user.name'].count()
#X2.groupby(  '_source.@timestamp')[  '_source.system.process.summary.running'].count()
#X2[  '_source.system.process.summary.running'].count()
#X2('_source.system.process.summary.running').[  '_source.user.name'].count()
#X2[ '_source.@timestamp'].nunique()
#CPU=X2.loc[X2[ '_source.event.dataset']=='system.cpu']
#X2.groupby( '_source.system.cpu.softirq.pct')[ '_source.system.cpu.softirq.pct'].count()
#CPU['_source.system.cpu.idle.pct'].describe()
#CPU['_source.system.cpu.idle.pct'].count(0)
#plt.hist(CPU['_source.system.cpu.idle.pct'])
#X2[ '_source.system.uptime.duration.ms'].describe()
#plt.hist(X2[ '_source.system.memory.actual.used.bytes'])
#X2[ '_source.system.uptime.duration.ms'].nunique()
#X2[ '_source.tags'].head()
#plt.hist(X2['_source.system.process.cpu.total.pct'])
#X2[[ '_source.system.process.cpu.total.pct','_source.system.cpu.system.pct', '_source.system.cpu.user.pct']].describe()
# '_source.system.cpu.system.pct', '_source.system.cpu.user.pct'
#X2[['_source.system.cpu.system.pct', '_source.system.cpu.user.pct','_source.system.cpu.total.pct']].head()

#### aggregating data 1 minute window
X0['@timestamp'].dtypes()
X0['@timestamp'] = pd.to_datetime(X0['@timestamp'])
X0['host_fingerprint']=X0['host.architecture']+'_'+X0['host.os.kernel']+'_'+X0['host.os.name']
import datetime
from sklearn.preprocessing import OneHotEncoder as oh

get_max = lambda x: x.value_counts(dropna=True); get_max.__name__ = "most frequent"


def moving_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def percentile_90(x):
        return x.quantile(0.9)
def percentile_75(x):
        return x.quantile(0.75)
def percentile_25(x):
        return x.quantile(0.25)
def percentile_10(x):
        return x.quantile(0.1)
def iqr(x):
    return x.quantile(0.75)-x.quantile(0.25)
def scatter(x):
    if x.mean()!=0:
        scatter=(x.mean()-x.median())/x.mean()
    else:
        
        'mean 0'
    return scatter

    

#def percentile(n):
#    def percentile_(x):
#        return x.quantile(n)
#    percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
#    return percentile_


# X0['_source.system.cpu.idle.pct']
    
# create list of functions for aggregation 
    

stats_list=['sum', 'mean','std','median',percentile_25,percentile_75,iqr,percentile_10,percentile_90]
col_order=['@timestamp','event.duration','system.cpu.softirq.pct','system.cpu.user.pct','system.cpu.total.pct',
           'system.cpu.irq.pct','system.cpu.cores','system.cpu.nice.pct','system.cpu.idle.pct','system.cpu.system.pct',
           'system.cpu.steal.pct','system.cpu.iowait.pct','system.memory.total','system.memory.used.bytes',
           'system.memory.used.pct','system.memory.actual.used.pct','system.memory.actual.used.bytes',
           'system.memory.actual.free','system.memory.swap.total','system.memory.swap.used.bytes','system.memory.swap.used.pct',
           'system.memory.swap.free','system.memory.free','system.process.cpu.total.value','system.process.cpu.total.pct',
           'system.process.cpu.total.norm.pct','system.process.memory.rss.pct','system.process.memory.rss.bytes',
           'system.process.memory.size','system.process.memory.share','process.pgid','process.pid','process.ppid',
           'system.socket.summary.tcp.all.established','system.socket.summary.tcp.all.close_wait',
           'system.socket.summary.tcp.all.listening','system.socket.summary.tcp.all.count','system.socket.summary.tcp.all.time_wait',
           'system.socket.summary.udp.all.count','system.socket.summary.all.listening','system.socket.summary.all.count',
           'system.network.out.errors','system.network.out.bytes','system.network.out.packets','system.network.out.dropped',
           'system.network.in.errors','system.network.in.bytes','system.network.in.packets','system.network.in.dropped',
           'system.process.summary.total','system.process.summary.zombie','system.process.summary.stopped',
           'system.process.summary.sleeping','system.process.summary.idle','system.process.summary.dead',
           'system.process.summary.unknown','system.process.summary.running','system.filesystem.total','system.filesystem.files',
           'system.filesystem.free_files','system.filesystem.available','system.filesystem.used.pct',
           'system.filesystem.used.bytes','system.filesystem.free','system.fsstat.total_files','system.fsstat.total_size.used',
           'system.fsstat.total_size.total','system.fsstat.total_size.free','system.fsstat.count','system.uptime.duration.ms',
           'metricset.period','event.dataset','metricset.name','system.process.cpu.start_time','system.process.state',
           'user.name','process.name','system.process.cmdline','process.args','system.network.name',
           'system.filesystem.mount_point','system.filesystem.type','system.filesystem.device_name','host_fingerprint']
X0=X[0:100]
X0=X0[col_order]
col_num=['@timestamp','event.duration','system.cpu.softirq.pct','system.cpu.user.pct','system.cpu.total.pct','system.cpu.irq.pct','system.cpu.cores','system.cpu.nice.pct','system.cpu.idle.pct','system.cpu.system.pct','system.cpu.steal.pct','system.cpu.iowait.pct','system.memory.total','system.memory.used.bytes','system.memory.used.pct','system.memory.actual.used.pct','system.memory.actual.used.bytes','system.memory.actual.free','system.memory.swap.total','system.memory.swap.used.bytes','system.memory.swap.used.pct','system.memory.swap.free','system.memory.free','system.process.cpu.total.value','system.process.cpu.total.pct','system.process.cpu.total.norm.pct','system.process.memory.rss.pct','system.process.memory.rss.bytes','system.process.memory.size','system.process.memory.share','process.pgid','process.pid','process.ppid','system.socket.summary.tcp.all.established','system.socket.summary.tcp.all.close_wait','system.socket.summary.tcp.all.listening','system.socket.summary.tcp.all.count','system.socket.summary.tcp.all.time_wait','system.socket.summary.udp.all.count','system.socket.summary.all.listening','system.socket.summary.all.count','system.network.out.errors','system.network.out.bytes','system.network.out.packets','system.network.out.dropped','system.network.in.errors','system.network.in.bytes','system.network.in.packets','system.network.in.dropped','system.process.summary.total','system.process.summary.zombie','system.process.summary.stopped','system.process.summary.sleeping','system.process.summary.idle','system.process.summary.dead','system.process.summary.unknown','system.process.summary.running','system.filesystem.total','system.filesystem.files','system.filesystem.free_files','system.filesystem.available','system.filesystem.used.pct','system.filesystem.used.bytes','system.filesystem.free','system.fsstat.total_files','system.fsstat.total_size.used','system.fsstat.total_size.total','system.fsstat.total_size.free','system.fsstat.count','system.uptime.duration.ms','metricset.period']
col_category=['@timestamp','event.dataset','metricset.name','system.process.cpu.start_time','system.process.state','user.name','process.name','system.process.cmdline','process.args','system.network.name','system.filesystem.mount_point','system.filesystem.type','system.filesystem.device_name','host_fingerprint']
col_category=['@timestamp','metricset.name']#,'event.dataset']#,'user.name','process.name','system.network.name','host_fingerprint']
col_category=['@timestamp','event.dataset']#,'metricset.name','user.name','process.name','system.network.name','host_fingerprint']
df_num=X0[col_num].set_index('@timestamp').resample('T').agg(stats_list)
df_cat=X0[col_category].set_index('@timestamp').resample('T').agg('mode')
df_category=X0[col_category].set_index('@timestamp').resample('T').agg([get_max])
df_category=X0[col_category].set_index('@timestamp').resample('T').agg([get_max])
len(X0['event.dataset'])
X_temp=X0[col_category]
X_temp = X_temp.set_index(pd.DatetimeIndex(X_temp['@timestamp']))
X_temp['@timestamp'] = pd.to_datetime(X_temp['@timestamp'])
df_category_numeric['event.dataset']=pd.factorize(X_temp['event.dataset'])[0]
df_category_numeric=df_category_numeric.set_index(pd.DatetimeIndex(X_temp['@timestamp']))
df_category_numeric['event.dataset'].resample('T').agg([get_max])
X_temp[col_category].set_index('@timestamp').resample('T').agg(['mode'])
X_temp[col_category].mode()

X_temp1=X_temp.set_index('@timestamp')
X_temp.set_index('@timestamp').resample('T').agg([datetime64]).bfill().ffill()

X_temp[X_temp.columns[1]]
df_category_numeric_index=pd.DataFrame()
df_category_numeric=pd.DataFrame()
for column in X_temp:
      df_category_numeric_index[column]=pd.factorize(X_temp[column])
      df_category_numeric[column]=pd.factorize(X_temp[column])[0]
df_category_numeric=df_category_numeric.set_index(df_category.index)
df = pd.concat([df_category_numeric, df_num], axis=1)
df_category.head()


df_category[[1]
df_category_new=df_category
pd.DataFrame()
df_category_new = df_category.stack()
df_categoty_new = pd.Series(df_category_new.factorize()[0], index=df_category_new.index).unstack()
print (df_categoty_new.stack().rank(method='dense').unstack())

vals = df_category.stack().drop_duplicates().values
b = [x for x in df_categoty_new.stack().drop_duplicates().rank(method='dense')]
[x for x in df_categoty_new.stack().drop_duplicates().rank(method='dense')]

d1 = dict(zip(b, vals))
print (d1)
print (df_categoty_new.stack().map(d1).unstack())


df_categoty_new_index
vals = df_categoty_new_index.stack().drop_duplicates().values
b = [x for x in df_categoty_new_index.stack().drop_duplicates().rank(method='dense')]

d1 = dict(zip(b, vals))
print (d1)
d.factorize(df_category)
df_category.rename(columns={"'_source.event.dataset', 'most frequent'":"event.dataset"}, errors="raise")
df_category.columns=['event.dataset','metricset.name','user.name','process.name','network.name','host_fingerprint']
 "_source.metricset.name', 'most frequent": "metricset.name", "_source.user.name', 'most frequent": "user.name","_source.process.name', 'most frequent":"process.name","_source.system.network.name', 'most frequent":"network.name","host_fingerprint', 'most frequent":"host_fingerprint"
df_category2 = pd.DataFrame({'data': df_category.unique(), 'data_new':range(len(data.data.unique()))})# create a temporary dataframe 
df_category = df_category.merge(df_category2, on='data', how='left')# Now merge it by assigning different values to different strings.

X0.iloc[:,0:72]
#reorder the DF seperating categorical from numeric data
X0['_source.@timestamp'][1]
X0['_index'].nunique()
get_max(X0['_source.event.duration'])
X0['_source.@timestamp'][1].isoweekday()
if X0['_source.@timestamp'][1].hour
X0['_source.system.filesystem.device_name'].unique()
X0['_source.process.args'].head()
df=X0.set_index('_source.@timestamp').resample('T').agg(stats_list)

df=X0.set_index('_source.@timestamp').resample('T').agg({'_source.system.cpu.idle.pct': stats_list})
df=X0.set_index('_source.@timestamp').resample('T').agg({'_source.system.cpu.idle.pct': stats_list,'_source.process.name':[get_max]})

df['day_night']=(df.index[15].hour<=19 and df.index[15].hour>=9)*1
df['weekend']=(df.index[15].isoweekday()==6 or df.index[15].isoweekday()==5)*1
df['hardware_fingerprint']=


print (stats_list[1])
stats_list[1]
r=print (stats_list[1])
temp_df=X0.agg({'_source.system.cpu.idle.pct': ['sum', 'mean','std','median',get_max,percentile(0.25),percentile(0.75),percentile(0.1),percentile(0.9),scatter]})
    
stats_list=['sum', 'mean','std','median']
,['percentile(0.25),percentile(0.75),(percentile(0.75)-percentile(0.25)),percentile(0.1),percentile(0.9),pizor']]
other_f="percentile(0.9)"+",pizor"

stats_list=['sum',other_f]

stats_list=['sum', 'mean','std','median',percentile(0.25),percentile(0.75),(percentile(0.75)-percentile(0.25)),percentile(0.1),percentile(0.9),pizor]
X0['_source.system.cpu.idle.pct'].quantile(0.1)
X0['_source.system.cpu.idle.pct'].pizor()
X0['_source.system.memory.total'].count()
rolling_mean = X0.y.rolling(window=20).mean()
list(X0.columns)
X0.dtypes
'_source.@timestamp'

X0['_source.host.architecture'].isna().sum()
X0['_source.host.os.kernel'].isna().sum()
X0['_source.host.os.name'].isna().sum()
get_max(X0['_source.process.name'])
X0['_source.process.name'].head()
X0.groupby('_source.process.name').count()
X0['_source.process.name'].value_counts(dropna=True).index[0]
X0[['_source.host.architecture','_source.host.os.kernel','_source.host.os.name']].apply(isnan)
#X0['hardware_fingerprint']= X0['_source.system.cpu.cores']+'_'+X0['_source.system.filesystem.total']+'_'+X0['_source.system.fsstat.total_size.total']+'_'+X0['_source.system.memory.total']

df=X0.set_index('_source.@timestamp').resample('T').agg({'_source.system.cpu.idle.pct': ['sum', 'mean'],'_source.process.name':[get_max]})

import numpy as np
X0['_source.system.cpu.idle.pct'].agg([np.sum, np.median, percentile(0.1)])
temp=X0.groupby(['_score', pd.Grouper(key='_source.@timestamp', freq='T')]).agg({'_source.system.memory.total':['sum']})
X0.groupby(['_score', pd.Grouper(key='_source.@timestamp', freq='A-DEC')]).agg({'_source.system.memory.total':['sum']})
pd.Grouper(key='_source.@timestamp', freq='A-DEC')]).agg({'_source.system.memory.total':['sum']})
pd.Grouper(key=X0['_source.@timestamp'], freq='A-DEC')['_source.system.memory.total'].sum()
pd.Grouper(key='_source.@timestamp', freq='M').agg('_source.system.memory.total'=sum())
X_count=
X_mea




#import bisect
#import collections
#import pandas as pd
#import pandas.TimeGrouper as TG
#pd.Tim
#X6.index = pd.to_datetime(X6.index)
#X7.index = pd.to_datetime(X7.index)
#X6.groupby(pd.TimeGrouper(freq='60Min'))
#X_new=X7.groupby(pd.Grouper(key='_source.system.memory.used.bytes', freq='60s'))
#X=X6.groupby([X6.index.minute]).sum()
#X=X6.groupby([X6.index.minute]).sum()
#X=X6.groupby([X6['_source.system.memory.used.bytes'],pd.TimeGrouper(freq='Min')])
#X_sum=X6.groupby(pd.TimeGrouper('1Min')).sum() 
#X_count=X6.groupby(pd.TimeGrouper('1Min')).count() 
#X_max=X6.groupby(pd.TimeGrouper('1Min')).max() 
#X_min=X6.groupby(pd.TimeGrouper('1Min')).min() 
#X_pct10=X6.groupby(pd.TimeGrouper('1Min')).quantile(.1) 
#X_pct25=X6.groupby(pd.TimeGrouper('1Min')).quantile(.25) 
#X_median=X6.groupby(pd.TimeGrouper('1Min')).median() 
#X_pct75=X6.groupby(pd.TimeGrouper('1Min')).quantile(.75) 
#X_pct90=X6.groupby(pd.TimeGrouper('1Min')).quantile(.9) 
#X_mean_=X6.groupby(pd.TimeGrouper('1Min')).mean() 
#list(X6.columns)
#X6['_source.system.memory.swap.free'].mean()
## model

train=df.dropna(1)
train=df.loc[:, (df != df.iloc[0]).any()]
train=train.dropna(1)

train=df.dropna(1)
train[train.isna()==True]
clf=IF(behaviour='new',max_samples=100,random_state=1,contamination='auto')

preds=clf.fit_predict(train)
list(preds).count(-1)
list(preds)


pickle.dump(clf,open('pkl_file','wb'))

# pca the X_train
plt.scatter(random_data2[:,:1],random_data2[:,1:2],)
random_data2[:,:1].shape
random_data2[:,:2].shape
preds2.shape
print(random_data2[:,:0])
