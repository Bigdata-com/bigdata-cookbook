from scipy.ndimage import gaussian_filter1d
import pandas as pd

def volume_z_score(x):
    mean = x.mean()
    std = x.std()
    if std == 0:
        return 0
    else:
        zscore = (x - mean) / std

    # Apply smoothing using Gaussian filter
    smoothed_data = gaussian_filter1d(zscore.fillna(0).values, sigma=2)
    return pd.Series(smoothed_data, index=x.index)

import numpy as np
def exponential_smoothing(x,decay):
    epw = np.exp(-(np.arange(len(x),0,-1) - 1) / decay)
    return np.sum(x*epw)

def create_full_grid_indicators(df,  start_date, end_date, smoothed=False):
    from itertools import product
    df_temp = df.groupby(["Date", "Entity"]).agg({"Bigdata Sentiment": "mean", 'Document ID':'nunique'}).reset_index()
    if 'Entity Sentiment' in df.columns:
        df_temp2 = df.groupby(["Date", "Entity"]).agg({'Entity Sentiment':'mean', 'Entity Text Sentiment':'mean'}).reset_index()
        df_temp = df_temp.merge(df_temp2, how='left', on=['Date', 'Entity'])

    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    fullgrid = pd.DataFrame(list(product(pd.date_range(start=start_date, end=end_date, freq='D'), df['Entity'].unique())), columns=['Date', 'Entity'])

    daily_sentiment = pd.merge(fullgrid, df_temp, on=['Date', 'Entity'], how='left')
    daily_sentiment['Avg_Sentiment'] = daily_sentiment['Bigdata Sentiment'].fillna(0)

    daily_sentiment['Sent_Rolling_30Days'] = daily_sentiment.groupby('Entity')['Avg_Sentiment'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())

    daily_sentiment['Sent_Rolling_90Days'] = daily_sentiment.groupby('Entity')['Avg_Sentiment'].transform(lambda x: x.rolling(window=90, min_periods=1).sum())

    daily_sentiment['Volume'] = daily_sentiment['Document ID'].fillna(0)
    daily_sentiment['volume_zscore'] = daily_sentiment.groupby('Entity')['Volume'].transform(lambda x: volume_z_score(x))

    if 'Entity Sentiment' in daily_sentiment.columns:
        daily_sentiment['Avg_Entity_Sentiment'] = daily_sentiment['Entity Sentiment'].fillna(0)
        daily_sentiment['Avg_Entity_Text_Sentiment'] = daily_sentiment['Entity Text Sentiment'].fillna(0)

    if smoothed:
        daily_sentiment['avg_sent_smoothed'] = daily_sentiment.groupby('Entity')['Avg_Sentiment'].transform(lambda x: x.rolling(window=180, min_periods=1).apply(exponential_smoothing, kwargs={'decay': 30}))
        if 'Entity Sentiment' in daily_sentiment.columns:
            daily_sentiment['avg_entitysent_smoothed'] = daily_sentiment.groupby('Entity')['Avg_Entity_Sentiment'].transform(lambda x: x.rolling(window=180, min_periods=1).apply(exponential_smoothing, kwargs={'decay': 90}))
            daily_sentiment['avg_entitytextsent_smoothed'] = daily_sentiment.groupby('Entity')['Avg_Entity_Text_Sentiment'].transform(lambda x: x.rolling(window=180, min_periods=1).apply(exponential_smoothing, kwargs={'decay': 90}))

    daily_sentiment = daily_sentiment.sort_values(by=['Entity', 'Date'])

    return daily_sentiment