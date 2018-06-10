
# Tweets Polarity Analysis

In this assignment, sentiment analysis of the Twitter activity of various news oulets has been performed, and the findings are presented in graphs. The news outlet selected for tweets analysis are __BBC, CBS, CNN, Fox, and New York times__


```
#import dependencies
import pandas as pd
import numpy as np
import tweepy
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from datetime import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

```


```
# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```
target_users = ("@nytimes","@BBCWorld","@CNN", "@CBSNews","@FoxNews")
sentiment_compound=[]
for user in target_users:
    counter=0
    oldest_tweet=None
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    for x in range(5):
        public_tweets = api.user_timeline(user, max_id=oldest_tweet)
        for tweet in public_tweets:
            results = analyzer.polarity_scores(tweet['text'])
            compound_list.append(results["compound"])
            positive_list.append(results["pos"])
            neutral_list.append(results["neu"])
            negative_list.append(results["neg"])
            time_stamp=counter+1
        oldest_tweet=tweet['id']-1 
        #print(f"{counter} {tweet['text']}")
    sentiment_dict={user: compound_list}
    sentiment_compound.append(sentiment_dict)
#     comp_series=pd.Series(sentiment_dict)
#     print(comp_series)
    #df_comp.append(comp_series,ignore_index=False)
    
            
```


```
print(sentiment_compound)
```

    [{'@nytimes': [0.8074, -0.4019, -0.6705, -0.128, 0.8225, 0.2023, 0.0, -0.296, 0.5859, 0.1901, 0.6486, -0.2411, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6249, 0.4019, -0.2732, -0.7351, 0.8126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.128, 0.0, 0.0, -0.0772, 0.4767, 0.0, 0.2732, -0.5994, 0.743, 0.3612, 0.3612, -0.7269, 0.2023, 0.6369, 0.7269, -0.875, -0.6705, -0.0258, 0.5106, 0.0, 0.6369, -0.296, 0.0, 0.4574, 0.7906, 0.0, 0.5859, -0.296, -0.296, 0.0, -0.3182, -0.8074, 0.0, 0.3612, 0.4404, 0.0, 0.6369, 0.25, 0.5267, -0.1531, 0.6124, 0.5994, -0.4215, 0.0, 0.5106, -0.7579, -0.6249, 0.8097, 0.0, -0.25, 0.7184, 0.0, 0.357, 0.5859, 0.2732, 0.0, -0.2023, -0.34, -0.0387, -0.5574, 0.1027, 0.2263, 0.0, -0.1027, 0.5719, 0.3182, 0.7906, 0.0, 0.0, 0.296, 0.0, 0.0, 0.0772]}, {'@BBCWorld': [0.0, 0.3309, -0.5106, 0.5423, -0.6486, 0.0258, 0.3612, -0.8225, 0.0, 0.5719, 0.5423, 0.5859, 0.0, -0.802, 0.0, -0.4215, 0.0, -0.7184, 0.5859, 0.0, 0.1027, 0.0, 0.0, 0.3612, 0.0, 0.0, 0.0, 0.3612, 0.3182, 0.0, -0.1027, -0.6369, 0.0, 0.0, 0.0, 0.0, -0.4939, 0.3612, 0.4277, -0.3612, -0.6705, 0.0, 0.0, 0.0, -0.6705, 0.0258, -0.2732, 0.0, 0.0, 0.4939, 0.2263, 0.0, 0.0, 0.0, 0.0, 0.4003, 0.7506, -0.0258, 0.0, -0.4767, 0.5859, 0.0, -0.3182, -0.9371, 0.2263, 0.2023, -0.6486, -0.6486, 0.7407, 0.0, 0.0, -0.4939, 0.0258, -0.2263, 0.0, 0.0, 0.0, 0.0, -0.4767, 0.0258, -0.4404, 0.0, 0.3612, -0.5994, -0.4019, 0.0, -0.3818, 0.5423, 0.0964, 0.0, 0.2023, 0.0, 0.0, 0.3182, -0.1531, 0.3182, -0.5267, 0.0, 0.2732, 0.0]}, {'@CNN': [0.6124, -0.5994, -0.296, 0.8176, 0.765, 0.5859, -0.6486, 0.2732, -0.5574, -0.296, 0.5927, -0.1027, 0.0, -0.4767, 0.0, 0.6705, 0.6486, 0.0, 0.539, 0.5859, 0.3612, -0.7269, -0.6486, -0.7506, 0.0, 0.3291, 0.0, 0.0, 0.4215, 0.4215, 0.765, 0.0, 0.0, 0.0, 0.2732, -0.4019, -0.7506, -0.5994, 0.4404, 0.7717, 0.0, 0.2023, 0.1779, -0.296, 0.0, -0.4585, -0.0772, 0.6597, 0.2732, 0.539, -0.3412, 0.5574, 0.0, 0.0, 0.34, -0.3818, 0.0, 0.5719, 0.5859, 0.0, 0.0, 0.6124, 0.3818, 0.4939, 0.4576, -0.5994, 0.0, -0.296, 0.3612, 0.765, -0.4588, 0.0, 0.0, 0.0, 0.4019, 0.5859, 0.0, 0.0, -0.6124, 0.6369, 0.3612, 0.0, 0.2732, 0.0, 0.0, -0.2732, 0.0, 0.0, 0.2263, 0.0, 0.8519, 0.7351, 0.0258, 0.4215, -0.3597, 0.0, 0.0, 0.5859, 0.8225, 0.0]}, {'@CBSNews': [0.0, -0.7351, -0.0772, -0.5859, -0.1779, 0.0, -0.1154, 0.8176, 0.296, 0.3818, -0.6705, -0.4767, 0.5719, -0.5216, -0.6369, 0.2755, -0.3182, -0.4939, 0.6369, -0.5106, -0.3182, -0.4019, 0.1779, 0.0, 0.765, 0.0, 0.0, 0.0, -0.0772, 0.296, 0.2732, -0.1154, 0.4019, 0.0, 0.5719, 0.0, -0.7096, -0.0772, 0.0, 0.0, 0.0, 0.6597, 0.8214, 0.5423, 0.3818, 0.0, -0.7269, -0.5994, -0.2263, 0.5106, -0.5106, -0.0258, -0.8481, -0.6808, 0.0, 0.0, -0.7579, -0.3182, 0.5574, -0.5994, 0.5859, 0.6705, -0.34, 0.0, -0.1779, -0.8126, -0.7096, -0.6486, -0.0258, -0.5106, -0.8481, -0.6124, 0.0, 0.296, -0.7579, 0.5574, -0.5994, 0.5574, -0.5574, 0.5859, -0.34, 0.0, -0.1779, -0.7096, -0.6486, -0.0258, 0.1531, -0.6486, -0.8481, -0.6124, 0.0, 0.0, 0.4019, 0.8442, 0.6846, 0.1531, 0.5859, 0.431, -0.7319, 0.0]}, {'@FoxNews': [-0.296, 0.3612, -0.5413, 0.2263, 0.5719, -0.4019, 0.4019, 0.7003, 0.8238, 0.2755, 0.7096, -0.5859, 0.0, 0.3182, 0.0, 0.6597, 0.7906, 0.5919, 0.0, 0.0, 0.0, 0.7351, 0.0, 0.0, 0.128, -0.6115, 0.0, -0.7088, 0.0, -0.4767, 0.0772, 0.3818, 0.0, 0.4939, 0.0, 0.891, 0.3612, 0.0, -0.6486, 0.0, 0.0, -0.2584, 0.0, -0.4939, 0.6249, -0.1531, 0.3182, 0.1655, 0.3612, -0.6249, -0.4215, -0.4767, -0.1531, -0.296, 0.3612, -0.7906, 0.0, 0.1027, -0.5302, 0.5859, -0.6249, 0.4939, 0.3612, 0.4767, 0.0, 0.4404, 0.0, -0.4404, 0.0, 0.0, 0.0, -0.4588, 0.1779, -0.5413, 0.0, 0.3612, -0.1531, -0.8555, -0.4019, 0.0, -0.5106, 0.4767, 0.0, -0.6705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3612, 0.0, -0.2023, 0.0, 0.0, -0.6115, -0.5994, -0.4767, 0.5859, 0.0]}]
    


```
# initializing empty dataframe of 100 rows
indx=range(100)
df_comp=pd.DataFrame(index=indx)
# join in a single df
for x in range(len(target_users)):
    df_x=pd.DataFrame(sentiment_compound[x])
    df_comp=df_comp.join(df_x)
df_comp['tweets ago']=range(1,101)
df_comp.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>@nytimes</th>
      <th>@BBCWorld</th>
      <th>@CNN</th>
      <th>@CBSNews</th>
      <th>@FoxNews</th>
      <th>tweets ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8074</td>
      <td>0.0000</td>
      <td>0.6124</td>
      <td>0.0000</td>
      <td>-0.2960</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4019</td>
      <td>0.3309</td>
      <td>-0.5994</td>
      <td>-0.7351</td>
      <td>0.3612</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.6705</td>
      <td>-0.5106</td>
      <td>-0.2960</td>
      <td>-0.0772</td>
      <td>-0.5413</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.1280</td>
      <td>0.5423</td>
      <td>0.8176</td>
      <td>-0.5859</td>
      <td>0.2263</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8225</td>
      <td>-0.6486</td>
      <td>0.7650</td>
      <td>-0.1779</td>
      <td>0.5719</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```
x_val=df_comp['tweets ago']
size=75
ax1 = df_comp.plot(kind='scatter', x='tweets ago', y='@nytimes', label='@nytimes', color='y', s=size,linewidths=1,edgecolor='black')    
ax2 = df_comp.plot(kind='scatter', x='tweets ago', y='@BBCWorld', label='@BBCWorld', color='g', s=size,linewidths=1,edgecolor='black',ax=ax1)    
ax3 = df_comp.plot(kind='scatter', x='tweets ago', y='@CNN', label='@CNN', color='b', s=size,linewidths=1,edgecolor='black',ax=ax1)
ax4 = df_comp.plot(kind='scatter', x='tweets ago', y='@CBSNews', label='@CBSNews', color='c', s=size,linewidths=1,edgecolor='black',ax=ax1)
ax5 = df_comp.plot(kind='scatter', x='tweets ago', y='@FoxNews', label='@FoxNews', color='r', s=size,linewidths=1,edgecolor='black',ax=ax1)
# ax6 = df_comp.plot(kind='scatter', x='tweets ago', y='@Reuters', color='w', ax=ax1)

print(ax1 == ax2 == ax3 == ax4 == ax5)  # True
ax1.set_xlim(x_val.max(),x_val.min())
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
lgnd = plt.legend(fontsize="small", mode="Expanded", 
                  numpoints=1, scatterpoints=1, 
                  loc="upper left", bbox_to_anchor=(1,1), title="Media Sources", 
                  labelspacing=0.5)
now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M")
plt.title(f"Sentiment Analysis of Tweets ({now}) for Various News Channels",fontsize=25)

# plt.legend()
# plt.fig_size(20,8)
```

    True
    




    Text(0.5,1,'Sentiment Analysis of Tweets (2018-06-09 19:02) for Various News Channels')




![png](output_7_2.png)



```
df_comp_mean=df_comp.mean()
df_comp_mean=df_comp_mean.drop('tweets ago')
```


```
sns.set()
df_comp_mean.plot(kind='bar',color='g', figsize=(20,5))
plt.title(f"Average Sentiment Analysis of Tweets ({now}) for Various News Channels",fontsize=25)
plt.ylabel("Tweet Polarity")

```




    Text(0,0.5,'Tweet Polarity')




![png](output_9_1.png)

